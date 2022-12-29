from iphyre.simulator import IPHYRE

actions = {'hole': [[251, 127, 0.7000000000000005], [247, 174, 1.8333333333333317]],
           'fill': [[146, 122, 2.499999999999996], [300, 196, 3.5499999999999923]],
           'seesaw': [[456, 316, 4.999999999999988]],
           'angle': [[399, 301, 0.8833333333333344], [398, 350, 3.983333333333324]],
           }

for game in ['hole', 'fill', 'seesaw', 'angle']:
    demo = IPHYRE(game)
    demo.collect_seq_data(save_path='../data/seq_data/', act_lists=[actions[game]], fps=20)
    