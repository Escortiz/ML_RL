import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space): 
        super().__init__()
        self.state_space = state_space # how many number define a space
        self.action_space = action_space # how many numbers define an action
        self.hidden = 64
        self.tanh = torch.nn.Tanh() #active function hiperbolic tangent

        """
            Actor network
            Arquitectura del actor
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)  #capa de entrada: toma el espacio y produce salidas
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)  
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)  #capa de salida: produce la media de la distribucion de acciones

        # Learned standard deviation for exploration at training time (log-std per action)
        # We parameterize std as exp(log_std) for numerical stability.
        init_sigma = 0.5
        self.log_std = torch.nn.Parameter(torch.full((self.action_space,), np.log(init_sigma), dtype=torch.float32))


        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_value = torch.nn.Linear(self.hidden, 1) # El crítico estima un solo valor V(s)


        self.init_weights()


    def init_weights(self): #pesos para las capas lineales 
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        # Build positive std from learnable log_std; expand to batch size if needed
        sigma = torch.exp(self.log_std)
        if sigma.dim() == 1 and action_mean.dim() == 2:
            sigma = sigma.unsqueeze(0).expand_as(action_mean)
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
        # TASK 3: forward in the critic network
        # TASK 3: forward in the critic network
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        value = self.fc3_critic_value(x_critic) # Estimación del valor del estado
        
        return normal_dist, value



class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device #dispositovo para el entrenamiento cpu o cuda(GPU)
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.gamma = 0.99 #factor de descuento para las recompensas futuras

        # Listas para almacenar la experiencia del agente (trayectorias)
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []


    def update_policy(self):
        """
        Actualiza la política usando el algoritmo REINFORCE.
        Esta función se llama típicamente al final de cada episodio.
        """
        # Convierte las listas de experiencia a tensores de PyTorch
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)


        #Limpia las listas de experiencia para el proximo episodio
        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        #
        # TASK 2:
        #   - compute discounted returns
        #   - compute policy gradient loss function given actions and returns
        #   - compute gradients and step the optimizer
        #
        # 1. Calcular retornos descontados (G_t)
        # Las recompensas ya están almacenadas en self.rewards y se convirtieron a un tensor 'rewards'
        # La función discount_rewards espera un tensor de recompensas por episodio.
        # Asumimos que los datos en self.rewards corresponden a un solo episodio.
        discounted_returns = discount_rewards(rewards, self.gamma)
        discounted_returns_norm = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-8) # Normalizar retornos para estabilidad
        
        # 2. Obtener valores de estado V(s_t) del crítico para los estados de la trayectoria
        # El método policy.forward() ahora devuelve (distribución, valor)
        # Necesitamos pasar los 'states' a través de la política para obtener las estimaciones de valor
        # No necesitamos la distribución de acciones aquí, solo los valores.
        _ , value_preds = self.policy(states) # El primer valor (distribución) no se usa aquí
        value_preds = value_preds.squeeze(-1) # Asegurar que la forma sea [N] y no [N, 1]



        # 3. Calcular ventajas A_t = G_t - V(s_t)
        # Usamos value_preds.detach() para que los gradientes de la pérdida del actor no afecten al crítico.
        advantages = discounted_returns_norm - value_preds.detach() 
    

          #
        # TASK 3:
        #   - compute boostrapped discounted return estimates
        #   - compute advantage terms
        #   - compute actor loss and critic loss
        #   - compute gradients and step the optimizer
        # 4. Calcular la función de pérdida del gradiente de la política ---- SOLO REINFORCE ----
        # o   Calcular pérdida del actor --- Baseline ------
        # La pérdida es - (sum_t (log_prob(a_t|s_t) * G_t))
        # Queremos maximizar la esperanza de los retornos descontados, por lo que minimizamos el negativo.
        """
        action_log_probs tiene las probabilidades de las acciones que el agente tomo en cada paso del episodio de acuerdo a la politica actual.
        discounted_return tiene los retornos descontados (Gt, las recompensas futuras con el factor de descuento) que tan bueno fue tomar la accion
        dentro del episodio
        """
        #------ Perdida solo con REINFORCE ---------
        #loss = - (action_log_probs * discounted_returns).sum() 

        # ------ perdiad con BASELINE ------
        actor_loss = -(action_log_probs * advantages).sum(dim=-1) # Minimizar la pérdida del actor (maximizar la esperanza de las ventajas)


        # 5. Calcular pérdida del crítico
        # Loss_critic = MSE(G_t, V(s_t))
        # El crítico intenta predecir los retornos descontados.
        critic_loss = F.mse_loss(value_preds, discounted_returns_norm) # Usar discounted_returns_norm si se normalizaron

        # 6. Combinar pérdidas
        # El coeficiente 0.5 para la pérdida del crítico es común.
        total_loss = actor_loss + 0.5 * critic_loss


        """
        Ajuste de parametros de la politica actual
        """
        self.optimizer.zero_grad() # Pone a cero los gradientes acumulados
        total_loss.backward() # Calcula los gradientes de la pérdida con respecto a los parámetros de la política
        self.optimizer.step() # Actualiza los parámetros de la política

        return total_loss.item(), actor_loss.item(), critic_loss.item() # Devuelve el valor de la pérdida (opcional, para logging)
   


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist,_ = self.policy(x)

        if evaluation:
            action = normal_dist.mean # Acción determinista
            action_log_prob = None # No se necesita para evaluación
        else:
            action = normal_dist.sample() # Acción estocástica
            action_log_prob = normal_dist.log_prob(action).sum(dim=-1)

        return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

