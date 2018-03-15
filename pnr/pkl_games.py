"""pkl_games.py

Usage:
	pkl_games.py.py <f_data_config>

Arguments:
	<f_data_config>  example ''pnrs.yaml''

Example:
	python pkl_games.py pnrs.yaml
"""
import config as CONFIG
from pnr.data.constant import data_dir, game_dir
import pandas as pd
import cPickle as pkl
from tqdm import tqdm
import os
from docopt import docopt
import yaml
from glob import glob


if __name__ == '__main__':
	arguments = docopt(__doc__)
	print ("...Docopt... ")
	print(arguments)
	print ("............\n")

	game_ids = []
	pnr_dir = '%s/%s' % (game_dir, 'pnr-annotations')

	f_data_config = '%s/%s' % (CONFIG.data.config.dir,arguments['<f_data_config>'])
	data_config = yaml.load(open(f_data_config, 'rb'))
	season_ids = data_config['data_config']['season_ids']
	for season_id in season_ids:
		season_path = os.path.join('%s/*.csv' % (pnr_dir))
		game_paths = glob(season_path)
		game_paths = filter(lambda s:s.split('/')[-1].startswith('raw-00'), game_paths)
		for game_path in tqdm(game_paths):
			# if not (game_path.split('/')[-1] in data_config['data_config']['game_ids']):
			#     continue
			print (game_path)
			game_id = game_path.split('/')[-1].split('-')[-1].split('.csv')[0]
			if game_id+'.pkl' in os.listdir(game_dir):
				game_ids.append(str(game_id))

	data_config['data_config']['game_ids'] = game_ids
	yaml.dump(data_config, open(f_data_config, 'w'))