# IPHYRE
This is a project to explore **I**nteractive **PHY**sical **RE**asoning.

[[Slide](https://docs.google.com/presentation/d/1OKlC00aDkR0MthhsVOM7LSOSjqk3Htxw/edit?usp=sharing&ouid=102328237315023582126&rtpof=true&sd=true)][[Intro](https://docs.google.com/document/d/1Z3e2mT0AolBsJ7iLIp-K531QlIDqqAXhijRO1hB4Olw/edit?usp=sharing)][[Game](https://www.youtube.com/watch?v=Ko2ZIc9YypY)]

[[pymunk github](https://github.com/viblo/pymunk)]

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
Please customize different games in the file `game_paras.py`, which contains the vertices of the blocks and positions and radiuses of the balls. See the following structure as an example:
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
You can run `run_game.py` by specifying `game_name` and `mode` to 'play', 'simulate' or 'collect' with certain game:
```angular2html
python run_game.py hinder play
```
* **Play:** play the game in a UI with keyboard signals to eliminate blocks.
* **Simulate:** simulate the game with the specified actions and present in a UI.
* **Collect data:** run the simulator to get data of states without a UI.

Run `play_all.sh` to play all games.