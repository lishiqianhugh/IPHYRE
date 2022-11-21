# IPHYRE
This is a project to explore **I**nteractive **PHY**sical **RE**asoning.

<p align="left">
    <a href='https://docs.google.com/presentation/d/1OKlC00aDkR0MthhsVOM7LSOSjqk3Htxw/edit?usp=sharing&ouid=102328237315023582126&rtpof=true&sd=true'>
      <img src='https://img.shields.io/badge/Slide-.PPTX-green?style=plastic&logo=Google%20chrome&logoColor=green' alt='Slide'>
    </a>
    <a href='https://docs.google.com/document/d/1Z3e2mT0AolBsJ7iLIp-K531QlIDqqAXhijRO1hB4Olw/edit?usp=sharing'>
      <img src='https://img.shields.io/badge/Intro-.DOCS-purple?style=plastic&logo=Google%20chrome&logoColor=purple' alt='Intro'>
    </a>
    <a href='https://www.youtube.com/watch?v=Ko2ZIc9YypY'>
      <img src='https://img.shields.io/badge/Games-YouTube-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='YouTube Games'>
    </a>
     <a href='https://lishiqianhugh.github.io/IPHYRE/'>
      <img src='https://img.shields.io/badge/Games-Page-yellow?style=plastic&logo=Google%20chrome&logoColor=yellow' alt='Games'>
    </a>
    </a>
     <a href='https://github.com/viblo/pymunk'>
      <img src='https://img.shields.io/badge/Pymunk-github-red?style=plastic&logo=Google%20chrome&logoColor=red' alt='Pymunk'>
    </a>
</p>


## Getting start
Run the command below to install all the packages required in this project.
```
pip install -r requirements.txt
```

How to get `requirements.txt`:
```angular2html
pip install pipreqs
pipreqs .
```

## Games
Please customize different games in the file `games/game_paras.py`, which contains the vertices of the blocks and positions and radiuses of the balls. See the following structure as an example:
```angular2html
'''
block: [[x1, y1],[x2, y2]] with a radius of 10
ball: [X, Y, Radius]
eli: 0/1 indicates whether the body can be eliminated
dynamic: 0/1 indicates whether the body can move under Newtonian laws

When designing, be careful about the following points:
1. the original point is at the left-top corner of the screen
2. when specifying the vertices of blocks, try to write from smaller numbers to larger numbers
3. the number of blocks equals to the number of eli and the number of dynamic.
'''

game_paras = {'support':
                  {'block': [[[200., 400.], [300., 400.]],
                             [[200., 500.], [300., 500.]]],
                   'ball': [[250., 340., 20.]],
                   'eli': [1, 1, 0],
                   'dynamic': [0, 0, 1],
                   },
              'hinder':
                  {'block': [[[200., 400.], [500., 400.]],
                             [[450., 300.], [450., 380.]],
                             [[500., 300.], [500., 380.]],
                             [[200., 150.], [300., 200.]]],
                   'ball': [[250., 100., 20.]],
                   'eli': [0, 1, 1, 1, 0],
                   'dynamic': [0, 0, 0, 0, 1],
                   },
              }

```
You can run `games/simulator.py` by specifying `game_name` and `mode` to interact with certain game:
```angular2html
python games/simulator.py hinder play
```
See the game list [here](https://lishiqianhugh.github.io/IPHYRE/).

Mode List:
* **play:** play the game in a UI with mouse clicks to eliminate blocks.
* **simulate:** simulate the game with the specified actions and only return the final results without a UI.
* **simulate_vis:** simulate the game with the specified actions and present in a UI.
* **collect while play:** get raw data and player's successful and failed actions after playing with a UI.
* **collect data:** get raw data, actions and body properties of dynamic states without a UI.
* **collect initial data:** get images and body properties of only initial states without a UI.

Run `scripts/play_all.sh` to play all games.

Run `scripts/collect_while_play_all.sh` to collect data while playing all games.

Run `scripts/collect_initial_data_all.sh` to collect data in the initial scenes of all games.

Run `scripts/train_all.sh` to train all the four folds.



## API
### collect_data
#### Input: 
- action_list
- fps (maximum is 60)
#### Output content:
store in **data/{game_name}/{action_index}**

##### images: 
- game image for each frame
##### actions.npy: 
- the position and timestep of this specific sequence of actions
##### vectors.npy: 
 - vectors[0]: **shape**
    - 0 for ball and 1 for block
 - vectors[1:6]: **property**  
    - For ball: [1:3] is center position; [3:5] is padding; [5] is radius
    - For block: [1:5] is line position; [5] is radius
 - vectors[7]: **eli**(if the object can be eliminated, the value is 1 otherwise 0)
 - vectors[8]: **dynamic**(if the object is dynamic, the value is 1 otherwise 0)

