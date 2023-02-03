def n_step_forward(n_step, sim, viewer=None):
    if viewer is None:
        for _ in range(n_step):
            sim.step()

    else:
        for _ in range(n_step):
            sim.step()
            viewer.render()
