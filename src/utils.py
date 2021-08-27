import torch
import torch.nn.functional as F

from torchvision.utils import make_grid
import matplotlib.pyplot as plt



def cross_entropy_onehot(pred, target, device):
      return (- target * F.log_softmax(pred, dim=-1)).sum().to(device)
      
def onehot(label, classes, device):
  onehot = torch.zeros(1, classes, device=device)
  onehot.scatter_(1, torch.tensor([[label]]).to(device), 1)
  return onehot


def train(model, target_grad, dummy_input, dummy_target, optimizer, epochs, device, plot=False, img=None):
  model.train()
  # training loop
  iterations = []
  stack_list = []
  loss = None
  for epoch in range(epochs):
    def closure():
      optimizer.zero_grad()
      # generate dummy data for generating dummy gradients
      doutput = model(dummy_input.unsqueeze(dim=0).to(device))
      target = F.softmax(dummy_target, dim=-1).to(device)
      loss = cross_entropy_onehot
      loss = loss(doutput, target, device)
      # print(loss)
      dummy_grad = torch.autograd.grad(loss, model.parameters(), create_graph=True) # dummy gradient
      # compute the MSE between the true and dummy grad
      square_loss = 0
      for dg, tg in zip(dummy_grad, target_grad):
        square_loss += ((dg - tg).pow(2).sum())
      square_loss.backward()
      return square_loss
    loss = closure()
    print(loss)
    optimizer.step(closure)
    if closure() < loss:
          torch.save(model.state_dict(), "./checkpoints/model_state.pth")
    print(f"Epoch {epoch+1}, loss: {closure()}")
    if plot:
      if (epoch) % 10 == 0:
        stack_list.append(torch.clone(dummy_input))
        stack = torch.stack(list(reversed(list(reversed(stack_list)) + [img])))
        print(len(stack))
        # print(f"Epoch {epoch+1}, loss: {closure()}")
        iterations.append(epoch)
        # if dummy_input.cpu().shape[0] == 1:
        grid = make_grid(stack, nrow=len(stack), )
        plt.imshow(grid.cpu().detach().permute(1, 2, 0))
        # print(dummy_input.min())
        # else:
          # plt.imshow(dummy_input.cpu().detach().permute(1, 2, 0))
          # print(dummy_input)
        plt.show()
  return dummy_input, img