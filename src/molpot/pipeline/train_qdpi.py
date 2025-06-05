import molpot as mpot
import torch


def main(max_steps: int = 10000):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load QDpi dataset and prepare neighbor lists
    qdpi = mpot.dataset.QDpi(save_dir="data/qdpi", device=device)
    qdpi.add_process(mpot.process.NeighborList(cutoff=5.0))

    subset = torch.utils.data.Subset(qdpi, range(10000))
    train_size = int(0.8 * len(subset))
    val_size = len(subset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(subset, [train_size, val_size])

    train_dl = mpot.DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_dl = mpot.DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    # Build PiNet2 based potential
    pinet = mpot.potential.nnp.PiNet2(
        depth=4,
        basis_fn=mpot.potential.nnp.radial.GaussianRBF(10, 5.0),
        cutoff_fn=mpot.potential.nnp.cutoff.CosineCutoff(5.0),
        pi_nodes=[64, 64],
        ii_nodes=[64, 64, 64, 64],
        pp_nodes=[64, 64, 64, 64],
        activation=torch.nn.Tanh(),
    )
    e_readout = mpot.potential.nnp.readout.Atomwise(
        n_neurons=[64, 64, 1],
        in_keys=[("pinet", "p1")],
        out_keys=[("predicts", "energy")],
        reduce="sum",
    )
    f_readout = mpot.potential.nnp.readout.PairForce(
        in_keys=("predicts", "energy"),
        dx_key=mpot.alias.pair_diff,
        out_keys=("predicts", "forces"),
        create_graph=True,
    )
    potential = mpot.potential.PotentialSeq(pinet, e_readout, f_readout).to(device)

    optimizer = torch.optim.Adam(potential.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.994)

    loss_fn = mpot.Constraint()
    loss_fn.add(
        "energy_mse",
        torch.nn.MSELoss(),
        ("predicts", "energy"),
        mpot.alias.E,
        log=True,
    )
    loss_fn.add(
        "force_mse",
        torch.nn.MSELoss(),
        ("predicts", "forces"),
        mpot.alias.F,
    )

    trainer = mpot.PotentialTrainer(
        model=potential,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )
    trainer.compile()
    trainer.add_lr_scheduler(scheduler)
    trainer.add_checkpoint("checkpoints/qdpi")
    trainer.run(train_data=train_dl, max_steps=max_steps, eval_data=val_dl)


if __name__ == "__main__":
    main()