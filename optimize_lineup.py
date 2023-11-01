from itertools import combinations, permutations, product
import numpy as np


class Optimized_Lineups:
    def __init__(self, owner, data, optimize_col='z'):
        self.owner = owner
        self.optimize_col = optimize_col
        self.data = data.sort_values(optimize_col, ascending=False)
        self.num_hitter_positions = 14
        self.d = self.data[self.data['Owner']==owner][['Player','all_pos', 'z', 'type']].set_index('Player').to_dict(orient='index')      
        self.p_dict = {k:v for (k,v) in self.d.items() if 'p' in v['type']}
        self.h_dict = {k:v for (k,v) in self.d.items() if 'h' in v['type']}
        self.catchers = [k for k,v in self.h_dict.items() if 'C\'' in v['all_pos']]#.replace('[','').replace(']','').replace('\'','').split(", ")
        self.first_basemen = [k for k,v in self.h_dict.items() if '1B' in v['all_pos']]
        self.second_basemen = [k for k,v in self.h_dict.items() if '2B' in v['all_pos']]
        self.third_basemen = [k for k,v in self.h_dict.items() if '3B' in v['all_pos']]
        self.shortstops = [k for k,v in self.h_dict.items() if 'SS' in v['all_pos']]
        self.middles = [k for k,v in self.h_dict.items() if 'MI' in v['all_pos']]
        self.corners = [k for k,v in self.h_dict.items() if 'CI' in v['all_pos']]
        self.outfielders = [k for k,v in self.h_dict.items() if 'OF' in v['all_pos']]
        self.dhs = [k for k,v in self.h_dict.items() if 'DH' in v['all_pos']]
        self.ofs = [i for i in combinations(self.outfielders, 5)]
        self.dh2 = [i for i in combinations(self.dhs, 2)]
        
    def _make_pitcher_combos(self):
        self.pitcher_combos = [i for i in combinations(self.p_dict.keys(), 9)]
        self.pitcher_z_list = self._z_list(self.pitcher_combos, self.p_dict)
        self.pitcher_idx = np.nanargmax(self.pitcher_z_list)
        self.pitcher_optimized_z = self.pitcher_z_list[self.pitcher_idx]
        self.pitcher_optimized_lineup = self.pitcher_combos[self.pitcher_idx]
        return
    
    def _z_list(self, pos_combos, player_dict):
        sum_z = 0
        z_list = []
        for i in range(len(pos_combos)):
            for name in pos_combos[i]:
                #print(i, sum_z)
                sum_z += player_dict[name]['z']
            z_list.append(sum_z)
            sum_z = 0
        return z_list
    
    def _make_hitter_combos(self):
        position_combos = [i for i in product(self.catchers, self.first_basemen, self.second_basemen, self.shortstops, self.third_basemen, self.middles, self.corners, self.ofs)]
        max_z = -999
        max_z_idx = -1
        for num in range(len(position_combos)):
            list_z = sum([self.h_dict[name]['z'] for name in position_combos[num] if type(name)!=tuple]+[self.h_dict[item]['z'] for sublist in position_combos[num] for item in sublist if type(sublist)==tuple])
            if list_z > max_z and len(set(list(position_combos[num][0:-1])+list(position_combos[num][-1]))) == 12:
                    max_z = list_z
                    max_z_idx = num
        
        base_12 = list(set(list(position_combos[max_z_idx][0:-1])+list(position_combos[max_z_idx][-1])))
        self.hitter_optimized_lineup = list(position_combos[max_z_idx][0:-1])+list(position_combos[max_z_idx][-1])+[i for i in self.h_dict.keys() if i not in base_12][:2]
        self.hitter_optimized_z = self.data[self.data['Player'].isin(self.hitter_optimized_lineup)]['z'].sum()
        #_list = []
        #for num in range(len(inf)):
        #    _list.append([item for item in inf[num] if type(item)!=tuple]+[item for sublist in inf[num] for item in sublist if type(sublist)==tuple])
        
        #del(inf)
        #_list2 = [i for i in _list if len(set(i))==12]
        #del(_list)
        #inf2 = [i for i in product(_list2, self.dh2)]
        
        #del(_list2)
        #_list3 = []
        #for num in range(len(inf2)):
            #_list3.append([list(i) for i in inf2[num] if type(i)!=tuple][0]+[list(i) for i in inf2[num] if type(i)==tuple][0])
        #del(inf2)

        #self.hitter_combos = [i for i in _list3 if len(set(i))==self.num_hitter_positions]
        #print(len(self.hitter_combos))
        #self.hitter_z_list = self._z_list(self.hitter_combos, self.h_dict)
        #print(len(self.hitter_z_list))
        #self.hitter_idx = np.nanargmax(self.hitter_z_list)
        #self.hitter_optimized_z = self.hitter_z_list[self.hitter_idx]
        #self.hitter_optimized_lineup = self.hitter_combos[self.hitter_idx]
        return    
