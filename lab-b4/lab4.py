import argparse, os, torch
p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="ckpt.pt")
p.add_argument("--steps", type=int, default=2000)
p.add_argument("--stop-at", type=int, default=None)
a = p.parse_args()

torch.manual_seed(0)
X = torch.randn(1024, 10)
y = X @ torch.randn(10, 1) + 0.1 * torch.randn(1024, 1)
model = torch.nn.Linear(10, 1)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.5)

step = 0
if os.path.exists(a.ckpt):
    c = torch.load(a.ckpt, weights_only=False)
    model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
    sched.load_state_dict(c["sched"]); torch.set_rng_state(c["rng"])
    step = c["step"]
    print(f"resumed at step {step}")

while step < a.steps:
    idx = torch.randint(0, 1024, (64,))
    loss = torch.nn.functional.mse_loss(model(X[idx]), y[idx])
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    step += 1
    if step % 100 == 0 or step == a.stop_at:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "rng": torch.get_rng_state(),
                    "step": step}, a.ckpt + ".tmp")
        os.replace(a.ckpt + ".tmp", a.ckpt)
    if step == a.stop_at:
        print(f"stopping at {step}"); break
print("final weight sum:", model.weight.sum().item())
