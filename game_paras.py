'''
block: [[x1, y1],[x2, y2]] with a radius of 10
eli: 0/1 indicates whether the block can be eliminated
dynamic: 0/1 indicates whether the block can move under Newtonian laws
ball: [X, Y, Radius]
When designing, be careful about the following points:
1. the original point is at the left-top corner of the screen
2. when specifying the vertices of blocks, try to write from smaller numbers to larger numbers
3. the number of blocks equals to the number of eli and the number of dynamic.
'''

game_paras = {'support':
                  {'block': [[[200., 400.], [300., 400.]],
                             [[200., 500.], [300., 500.]],
                             [[400., 400.], [500., 400.]],
                             [[400., 500.], [500., 500.]]],
                   'eli': [1, 1, 1, 1],
                   'dynamic': [0, 0, 0, 0],
                   'ball': [[250., 340., 20.]]
                   },
              'hinder':
                  {'block': [[[200., 400.], [500., 400.]],
                             [[450., 300.], [450., 380.]],
                             [[500., 300.], [500., 380.]],
                             [[200., 150.], [300., 200.]]],
                   'eli': [0, 1, 1, 1],
                   'dynamic': [0, 0, 0, 0],
                   'ball': [[250., 100., 20.]]
                   },
              'direction':
                  {'block': [[[100., 400.], [500., 400.]],
                             [[100., 300.], [100., 380.]],
                             [[500., 300.], [500., 380.]],
                             [[100., 150.], [280., 250.]],
                             [[320., 250.], [500., 150.]]],
                   'eli': [0, 1, 1, 1, 1],
                   'dynamic': [0, 0, 0, 0, 0],
                   'ball': [[300., 100., 50.]]
                   },
              'hole':
                  {'block': [[[100., 400.], [250., 400.]],
                             [[350., 400.], [500., 400.]],
                             [[450., 300.], [450., 380.]],
                             [[500., 300.], [500., 380.]],
                             [[200., 150.], [300., 200.]],
                             [[250., 100.], [250., 150.]],
                             ],
                   'eli': [0, 0, 1, 1, 1, 1],
                   'dynamic': [0, 0, 0, 0, 0, 0],
                   'ball': [[220., 120., 20.]]
                   },
          #     'multi_balls':
          #         {'block': [[[250., 200.], [350., 200.]],
          #                    [[150., 400.], [230., 400.]],
          #                    [[370., 400.], [450., 400.]]],
          #          'eli': [1, 1, 1],
          #          'dynamic': [0, 0, 0],
          #          'ball': [[300., 130., 60.],
          #                   [190., 330., 60.],
          #                   [410., 330., 60.]]
          #          },
               'multi_balls':
                    {'block': [[[450., 400.], [550., 400.]],
                              [[350., 600.], [430., 600.]],
                              [[570., 600.], [650., 600.]]],
                    'eli': [1, 1, 1],
                    'dynamic': [0, 0, 0],
                    'ball': [[500., 330., 60.],
                              [390., 530., 60.],
                              [610., 530., 60.]]
                    },
               
              }