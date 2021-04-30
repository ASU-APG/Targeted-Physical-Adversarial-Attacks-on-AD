import torch
from policy.networks.actor_critic import A2CNet


class Agent():
    """
    Agent for testing
    """

    def __init__(self, img_stack, device):
        self.net = A2CNet(img_stack).float().to(device)

    def select_action(self, state, device):
        # state array contains values with in the range [-1, 0.9921875]
        state = torch.from_numpy(state.copy()).float().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        action = alpha / (alpha + beta)

        action = action.squeeze().cpu().numpy()
        return action

    def load_param(self, device):
        if device == torch.device('cpu'):
            self.net.load_state_dict(torch.load('policy/param/ppo_net_params.pkl', map_location='cpu'))
        else:
            self.net.load_state_dict(torch.load('policy/param/ppo_net_params.pkl'))
