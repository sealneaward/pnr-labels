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
	def __init__(self, moment, anno=None):
		self.quarter = moment[0]  # Hardcoded position for quarter in json
		self.game_clock = _convert_type(moment[2])  # Hardcoded position for game_clock in json
		self.game_clock_unix = int(moment[1])
		self.shot_clock = _convert_type(moment[3])  # Hardcoded position for shot_clock in json
		ball = moment[5][0]  # Hardcoded position for ball in json
		self.ball = Ball(ball)
		self.players = []

		players = moment[5][1:]  # Hardcoded position for players in json
		try:
			self.players = [Player(player, anno) for player in players]
		except PlayerNotFoundException:
			raise PNRException
		except TeamNotFoundException:
			raise TeamNotFoundException

		if len(self.players) != 10:
			raise MomentException("Not enough players")