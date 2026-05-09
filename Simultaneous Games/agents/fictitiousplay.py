from itertools import product
import numpy as np
from numpy import ndarray
from base.agent import Agent
from base.game import SimultaneousGame, AgentID

class FictitiousPlay(Agent):
    
    def __init__(self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None) -> None:
        super().__init__(game=game, agent=agent)
        np.random.seed(seed=seed)
        
        self.count: dict[AgentID, ndarray] = {}
        #
        # TODO: inicializar count con initial si no es None o, caso contrario, con valores random
        #
       
        # creamos un arreglo para cada agente, el cual cuenta cuantas veces se jugo cada accion
        for agent in game.agents:
            n = game.num_actions(agent)
            if initial is not None:
                self.count[agent] = np.array(initial[agent], dtype=float)
            else:
                self.count[agent] = np.random.random(n)

        self.learned_policy: dict[AgentID, ndarray] = {}
        #
        # TODO: inicializar learned_policy usando de count
        #
       
        # definimos la learned_policy como cuanto jugo cada accion sobre el total
        for agent in game.agents:
            self.learned_policy[agent] = self.count[agent] / np.sum(self.count[agent])

    def get_rewards(self) -> dict:
        # el agente solo ve los suyos propios
        g = self.game.clone()
        # lista de acciones por agente:
        agents_actions = list(map(lambda agent: list(g.action_iter(agent)), g.agents))
        rewards: dict[tuple, float] = {}
        #
        # TODO: calcular los rewards de agente para cada acción conjunta
        # Ayuda: usar product(*agents_actions) de itertools para iterar sobre agents_actions
        #
        
        # iteramos sobre todos las posibles combinaciones (agente/accion)
        for joint_action in product(*agents_actions):
            actions = dict(zip(g.agents, joint_action))
            # para cada combinacion jugamos y nos quedamos con la reward
            _, agent_rewards, _, _, _ = g.step(actions)
            rewards[joint_action] = agent_rewards[self.agent]
        
        return rewards
    
    def get_utility(self):
        # la utilidad es la esperanza de la recompenza
        rewards = self.get_rewards()
        utility = np.zeros(self.game.num_actions(self.agent))
        #
        # TODO: calcular la utilidad (valor) de cada acción de agente.
        # Ayuda: iterar sobre rewards para cada acción de agente
        #
        # para cada acción conjunta, calculamos qué tan probable es que los otros agentes
        # la jueguen (multiplicando sus probabilidades según learned_policy).
        # eso lo pesamos con el reward y lo sumamos a la utilidad de nuestra acción en esa jugada.
        # al final nos queda el valor esperado de cada acción propia
        agent_idx = self.game.agent_name_mapping[self.agent]
        for joint_action, reward in rewards.items():
            prob = 1.0
            for other, other_action in zip(self.game.agents, joint_action):
                if other != self.agent:
                    prob *= self.learned_policy[other][other_action]
            utility[joint_action[agent_idx]] += prob * reward
        return utility
    
    def bestresponse(self):
        a = None
        #
        # TODO: retornar la acción de mayor utilidad
        #
        # simplemente agarramos el índice de la acción con mayor utilidad esperada,
        # esa es la mejor respuesta dado lo que creemos que van a jugar los demás
        a = int(np.argmax(self.get_utility()))
        return a
     
    def update(self) -> None:
        actions = self.game.observe(self.agent)
        if actions is None:
            return
        for agent in self.game.agents:
            self.count[agent][actions[agent]] += 1
            self.learned_policy[agent] = self.count[agent] / np.sum(self.count[agent])

    def action(self):
        self.update()
        return self.bestresponse()
    
    def policy(self):
       return self.learned_policy[self.agent]
    