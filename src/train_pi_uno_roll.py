"""
MIT License - Copyright (c) 2026 Hongyi Guan
See LICENSE file for full license text
"""

"""
Physics-informed unrolled training of the UNO surrogate, with a grid search.
 
Extends the neuraloperator Trainer in two ways. (i) UnrolledTrainer applies the
model K times autoregressively inside a single optimisation step, feeding each
prediction back as the next input, while the matching ground truth is generated
on the fly by advancing the de-normalised input with the spectral solver. The
loss sums the H1 error over all K horizons, so the model is penalised for the
error growth that dominates long rollouts instead of for one-step accuracy
alone. (ii) BarrierLoss adds tan^2((c_pred - c_true) * pi/4), a term that is
mild for small errors but diverges as the mismatch approaches the full phase
span of 2, softly enforcing the physical bound c in [-1, 1]; its argument is
clamped so the penalty saturates rather than producing infinite gradients.
Gradients are clipped and the learning rate follows a cosine schedule.
Evaluation is deliberately left as one-step prediction so reported H1/L2 numbers
stay comparable with the baseline. run_grid_search sweeps barrier weight and K,
checkpointing the best model by validation H1 error.
"""

import math
import os
import itertools

import torch
import matplotlib
matplotlib.use("Agg")

from torch.utils.data import DataLoader

from neuralop.training import AdamW
from neuralop.training.training_state import save_training_state, load_training_state
from neuralop import H1Loss, LpLoss, Trainer

from CH_FFT import step_cahn_hilliard
from train_uno import build_uno_model, build_dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPS_CH = 0.1
A_CH = 2.0
DT_CH = 0.01
STRIDE = 10
L_DOMAIN = 2.0 * math.pi


def make_fft_constants(N_grid, dx, dt, eps, a, device):
    freqs = torch.fft.fftfreq(N_grid, d=dx, device=device, dtype=torch.float32) * 2.0 * torch.pi
    KY, KX = torch.meshgrid(freqs, freqs, indexing="ij")
    k2 = KX ** 2 + KY ** 2
    k4 = k2 ** 2
    denom = 1.0 + dt * (eps ** 2 * k4 + a * k2)
    return k2, denom


@torch.no_grad()
def fft_step_10_batched(c_batch, k2, denom):
    curr = c_batch
    for _ in range(STRIDE):
        curr = step_cahn_hilliard(curr, DT_CH, EPS_CH, A_CH, k2, denom).clone()
    return curr


class BarrierLoss(object):
    """tan^2 boundary-shift barrier between (c_pred, c_true), batch-summed.

    Inputs arrive in normalized space; we de-normalise both before
    applying the barrier because the barrier is geometric in c in
    [-1, +1] (the singularity at delta_c = +-2 only makes sense in
    physical scale). The argument is clamped to [-pi/2 + delta, pi/2 - delta]
    so the per-pixel loss is capped at cot^2(delta) ~ 100 and the
    gradient is zero past the clamp.
    """

    def __init__(self, weight=1e-2, delta=0.1, out_normalizer=None):
        self.weight = float(weight)
        self.delta = float(delta)
        self.z_max = 0.5 * math.pi - self.delta
        self.out_normalizer = out_normalizer

    def _denorm(self, t):
        if self.out_normalizer is None:
            return t
        return self.out_normalizer.inverse_transform(t)

    def __call__(self, y_pred, y_true):
        c_pred = self._denorm(y_pred)
        c_true = self._denorm(y_true)
        z = (c_pred - c_true) * (math.pi / 4.0)
        z = z.clamp(min=-self.z_max, max=self.z_max)
        cosz = torch.cos(z)
        per_pixel = 1.0 / (cosz * cosz) - 1.0
        per_sample = per_pixel.mean(dim=tuple(range(1, per_pixel.dim())))
        return self.weight * per_sample.sum()


class UnrolledTrainer(Trainer):
    """Trainer subclass that does K-step unrolled training for CH.

    Overrides train_one_batch to:
        (i)   run K consecutive model applications, feeding each output
              back as the next input (in normalised space);
        (ii)  generate K FFT targets on the fly from the ground-truth
              c_n (in physical space), renormalised for comparison;
        (iii) accumulate H1 + barrier_weight * Barrier at each of the K
              horizons.
    Evaluation is left untouched; the parent Trainer.eval_one_batch still
    runs one-step prediction so reported H1/L2 numbers stay comparable
    to the baseline runs.
    """

    def __init__(
        self,
        *args,
        K=3,
        h1_loss=None,
        barrier_loss=None,
        k2=None,
        denom=None,
        grad_clip=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.K = int(K)
        self.h1_loss = h1_loss
        self.barrier_loss = barrier_loss
        self.k2 = k2
        self.denom = denom
        self.grad_clip = grad_clip

    def _normalise_out(self, c_phys):
        out_norm = getattr(self.data_processor, "out_normalizer", None)
        if out_norm is None:
            return c_phys
        with torch.no_grad():
            return out_norm.transform(c_phys)

    def _denormalise_in(self, c_norm):
        in_norm = getattr(self.data_processor, "in_normalizer", None)
        if in_norm is None:
            return c_norm
        with torch.no_grad():
            return in_norm.inverse_transform(c_norm)

    def train_one_batch(self, idx, sample, training_loss):
        self.optimizer.zero_grad(set_to_none=True)

        sample = self.data_processor.preprocess(sample)
        x = sample["x"]
        y = sample["y"]

        if isinstance(y, torch.Tensor):
            self.n_samples += y.shape[0]
        else:
            self.n_samples += 1

        c_truth_phys = self._denormalise_in(x)
        curr_input = x
        total_loss = 0.0

        for k in range(self.K):
            pred_k = self.model(curr_input)
            c_truth_phys = fft_step_10_batched(c_truth_phys, self.k2, self.denom)
            y_k_norm = self._normalise_out(c_truth_phys)

            total_loss = total_loss + self.h1_loss(pred_k, y_k_norm)
            if self.barrier_loss is not None:
                total_loss = total_loss + self.barrier_loss(pred_k, y_k_norm)

            curr_input = pred_k

        return total_loss

    def train_one_epoch(self, epoch, train_loader, training_loss):
        from timeit import default_timer

        self.on_epoch_start(epoch)
        avg_loss = 0.0
        avg_lasso_loss = 0.0
        self.model.train()
        if self.data_processor:
            self.data_processor.train()
        t1 = default_timer()
        train_err = 0.0
        self.n_samples = 0

        for idx, sample in enumerate(train_loader):
            loss = self.train_one_batch(idx, sample, training_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()

            train_err += loss.item()
            with torch.no_grad():
                avg_loss += loss.item()
                if self.regularizer:
                    avg_lasso_loss += self.regularizer.loss

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(train_err)
        else:
            self.scheduler.step()

        epoch_train_time = default_timer() - t1
        train_err /= len(train_loader)
        avg_loss /= self.n_samples
        if self.regularizer:
            avg_lasso_loss /= self.n_samples
        else:
            avg_lasso_loss = None

        lr = None
        for pg in self.optimizer.param_groups:
            lr = pg["lr"]
        if self.verbose and epoch % self.eval_interval == 0:
            self.log_training(
                epoch=epoch,
                time=epoch_train_time,
                avg_loss=avg_loss,
                train_err=train_err,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr,
            )
        return train_err, avg_loss, avg_lasso_loss, epoch_train_time


def train_evaluate_pi_uno(
    hidden_channels,
    lr,
    weight_decay,
    barrier_weight,
    K,
    n_epochs,
    batch_size,
    resolution,
    save_dir,
):
    print(f"\n{'=' * 60}")
    print(
        f"Hidden: {hidden_channels}  LR: {lr}  WD: {weight_decay}  "
        f"barrier_w: {barrier_weight}  K: {K}"
    )
    print(f"Save dir: {save_dir}")
    print(f"{'=' * 60}")

    dataset = build_dataset(resolution, batch_size)
    train_loader = DataLoader(
        dataset.train_db, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    test_loaders = {
        resolution: DataLoader(
            dataset.test_dbs[resolution],
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
    }
    data_processor = dataset.data_processor.to(DEVICE)

    model = build_uno_model(hidden_channels).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    h1loss = H1Loss(d=2)
    l2loss = LpLoss(d=2, p=2)

    barrier_loss = None
    if barrier_weight > 0.0:
        barrier_loss = BarrierLoss(
            weight=barrier_weight,
            delta=0.1,
            out_normalizer=getattr(data_processor, "out_normalizer", None),
        )

    dx = L_DOMAIN / resolution
    k2, denom = make_fft_constants(resolution, dx, DT_CH, EPS_CH, A_CH, DEVICE)

    eval_losses = {"h1": h1loss, "l2": l2loss}

    os.makedirs(save_dir, exist_ok=True)

    trainer = UnrolledTrainer(
        model=model,
        n_epochs=n_epochs,
        device=DEVICE,
        data_processor=data_processor,
        wandb_log=False,
        eval_interval=10,
        use_distributed=False,
        verbose=False,
        K=K,
        h1_loss=h1loss,
        barrier_loss=barrier_loss,
        k2=k2,
        denom=denom,
        grad_clip=1.0,
    )

    target_metric = f"{resolution}_h1"

    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=None,
        training_loss=h1loss,
        eval_losses=eval_losses,
        save_best=target_metric,
        save_dir=save_dir,
    )

    final_dir = os.path.join(save_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    save_training_state(
        save_dir=final_dir,
        save_name="final",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=n_epochs,
    )

    model, _, _, _, best_epoch = load_training_state(
        save_dir=save_dir,
        save_name="best_model",
        model=model,
    )
    if best_epoch is not None:
        print(f"Best checkpoint was saved at epoch {best_epoch}.")

    final_eval_metrics = trainer.evaluate_all(
        epoch=n_epochs,
        eval_losses=eval_losses,
        test_loaders=test_loaders,
        eval_modes={},
    )
    best_h1_error = final_eval_metrics[target_metric]
    print(f"Best validation H1 error: {best_h1_error:.6f}")
    return best_h1_error


def run_grid_search():
    N_EPOCHS = 500
    BATCH_SIZE = 50
    RESOLUTION = 64
    BASE_CHECKPOINT_DIR = "./grid_search_pi_roll_checkpoints"

    grid = {
        "hidden_channels": [32],
        "lr": [4e-3],
        "weight_decay": [1.2e-4],
        "barrier_weight": [0.0, 1e-2],
        "K": [1, 3, 5],
    }

    keys, values = zip(*grid.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_overall_error = float("inf")
    best_overall_config = None
    best_model_dir = None

    for idx, config in enumerate(permutations):
        run_save_dir = os.path.join(
            BASE_CHECKPOINT_DIR,
            (
                f"run_{idx}_C{config['hidden_channels']}_LR{config['lr']}"
                f"_WD{config['weight_decay']}_B{config['barrier_weight']}"
                f"_K{config['K']}"
            ),
        )
        error = train_evaluate_pi_uno(
            hidden_channels=config["hidden_channels"],
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            barrier_weight=config["barrier_weight"],
            K=config["K"],
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            resolution=RESOLUTION,
            save_dir=run_save_dir,
        )
        if error < best_overall_error:
            best_overall_error = error
            best_overall_config = config
            best_model_dir = run_save_dir

    print(f"\n{'*' * 60}")
    print("Unrolled-training grid search complete.")
    print(f"Best configuration : {best_overall_config}")
    print(f"Best H1 error      : {best_overall_error:.6f}")
    print(f"Best run directory : {best_model_dir}")
    print(f"{'*' * 60}")


if __name__ == "__main__":
    run_grid_search()
