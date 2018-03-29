import pandas as pd

from Team import Team, TeamNotFoundException

class PlayerNotFoundException(Exception):
    pass

class Player:
    """A class for keeping info about the players"""
    def __init__(self, player, anno=None):
        try:
            self.team = Team(player[0])
        except TeamNotFoundException:
            raise TeamNotFoundException

        self.id = player[1]
        self.x = player[2] # the long axis
        self.y = player[3]

        if anno is None:
            self.color = self.team.color
        else:
            anno['ball_handler'] = int(anno['ball_handler'])
            anno['ball_defender'] = int(anno['ball_defender'])
            anno['screen_setter'] = int(anno['screen_setter'])
            anno['screen_defender'] = int(anno['screen_defender'])

            if (self.id != anno['ball_handler']) and (self.id != anno['ball_defender']) and (self.id != anno['screen_setter']) and (self.id != anno['screen_defender']):
                self.color = self.team.color
            elif (self.id == anno['ball_handler']):
                self.color = '#ff0000'
            elif (self.id == anno['ball_defender']):
                self.color = '#0000ff'
            elif (self.id == anno['screen_setter']):
                self.color = '#9900cc'
            elif (self.id == anno['screen_defender']):
                self.color = '#33cc33'
            else:
                self.color = '#663300'
