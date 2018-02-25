from Ball import Ball
from Player import Player, PlayerNotFoundException
from Team import TeamNotFoundException

class MomentException(Exception):
	pass

class PNRException(Exception):
	pass


def _convert_type(inp):
	if inp==None or inp=='':
		ret = 0.0
	else:
		ret = float(inp)
	return ret


class Moment:
	"""A class for keeping info about the moments"""
	def __init__(self, moment, pnr=False, anno=None, one_hot=False, one_hot_players=None):
		self.one_hot_players = one_hot_players
		self.quarter = moment[0]  # Hardcoded position for quarter in json
		self.game_clock = _convert_type(moment[2])  # Hardcoded position for game_clock in json
		self.game_clock_unix = int(moment[1])
		self.shot_clock = _convert_type(moment[3])  # Hardcoded position for shot_clock in json
		ball = moment[5][0]  # Hardcoded position for ball in json
		self.ball = Ball(ball)
		players = moment[5][1:]  # Hardcoded position for players in json
		if pnr:
			try:
				self.players = [Player(player, one_hot_players) for player in players]
			except PlayerNotFoundException:
				raise PNRException
		else:
			try:
				self.players = [Player(player) for player in players]
			except TeamNotFoundException:
				raise TeamNotFoundException

		if len(self.players) != 10:
			raise MomentException("Not enough players")

		if pnr:
			player_ids = []
			for player in self.players:
				player_ids.append(player.id)

			players = self.players
			self.players = [None] * 4
			for player in players:
				if player.id == anno['ball_player_id']:
					self.players[0] = player
				elif player.id == anno['screen_player_id']:
					self.players[1] = player
				elif player.id == anno['ball_def_player_id']:
					self.players[2] = player
				elif player.id == anno['screen_def_player_id']:
					self.players[3] = player

			if not (
				self.players[0] is not None and
				self.players[1] is not None and
				self.players[2] is not None and
				self.players[3] is not None
			):
				raise PNRException