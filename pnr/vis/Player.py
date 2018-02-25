from Team import Team, TeamNotFoundException

class PlayerNotFoundException(Exception):
    pass

class Player:
    """A class for keeping info about the players"""
    def __init__(self, player, one_hot_players=None):
        try:
            self.team = Team(player[0])
        except TeamNotFoundException:
            raise TeamNotFoundException
        self.id = player[1]
        self.x = player[2] # the long axis
        self.y = player[3]
        self.color = self.team.color
        self.one_hot_vector = None

        if one_hot_players is not None:
            try:
                self.one_hot_vector = one_hot_players[self.id]
            except Exception:
                raise PlayerNotFoundException("player not found in dict")