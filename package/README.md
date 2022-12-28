# IPHYRE
The `iphyre` package is built to explore interactive physical reasoning.

## Introduction
IPHYRE benchmark contains 40 interactive physics games composed mainly of gray blocks, black blocks, blue blacks, and red balls. These games share the same goal of getting the red balls into the abyss, which can be reached by eliminating gray blocks (the only ones that can be eliminated). Besides the fixed static blocks (gray and black), we add dynamic blue blocks moving under gravity, springs, and pendulum bars to the environment to extensively enrich the physical dynamics.

## API
Some useful APIs are provided in iphyre.
* **iphyre.games.GAMES:** Get the names of all the games.
* **iphyre.games.PARAS:** Get the design parameters of all the games.
* **iphyre.simulator.reset():** Reset the bodyies in the game and return the initial state.
* **iphyre.simulator.step():** Apply an action and forward a timestep. Return the next state, reward and done.
* **iphyre.simulator.simulate():** Simulate the game with the specified actions and only return the final results without a UI.
* **iphyre.simulator.simulate_vis():** Simulate the game with the specified actions and display in a UI.
* **iphyre.simulator.play():** Play the game in a UI with mouse clicks to eliminate blocks.
* **iphyre.simulator.collect_initial_data():** Save images and body properties of only initial states without a UI.
* **iphyre.simulator.collect_seq_data():** Save raw data, actions and body properties of the dynamic state sequence without a UI.
* **iphyre.simulator.collect_while_play():** Save player's actions and rewards after playing with a UI.
* **iphyre.simulator.get_action_space():** Get the central positions of the eliminable blocks with no action at the first place and the padding place.
* **iphyre.utils.generate_actions():** Random generate successful and failed actions of specifc numbers in one game.
* **iphyre.utils.play_all():** play all the games.
* **iphyre.utils.collect_initial_all():** Save images and body properties of initial states in all the games.
* **iphyre.utils.collect_seq_all():** Save raw data, actions and body properties of the dynamic state sequence in all the games.
* **iphyre.utils.collect_play_all():** Play and save player's actions and rewards in all the games.