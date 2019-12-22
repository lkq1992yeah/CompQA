import json


class Cell:
    def __init__(self, surface, pos, label=None, blank=False, ground=None):
        self.surface = surface
        self.pos = pos
        self.label = label
        self.blank = blank
        self.mentions = []
        self.ground = ground

class Mention:
    def __init__(self, surface, entity, cell=None, pos=None):
        self.surface = surface
        self.entity = entity
        self.cell = cell
        self.pos = pos
        self.feature = None
        self.foo = None

class Table:
    def __init__(self, size, cells):
        self.size = size
        self.cells = cells        

    def get_cell(self, row, col):
        return self.cells[row * size[1] + col]

    def get_row_context(self, pos):
        row = pos[0]
        col = pos[1]
        ret = []
        for idx in range(row * self.size[1] , row * self.size[1] + col):
            if not self.cells[idx].blank:
                ret.append(self.cells[idx])
        for idx in range(row * self.size[1] + col + 1 , (row + 1) * self.size[1]):
            if not self.cells[idx].blank:
                ret.append(self.cells[idx])
        return ret

    def get_col_context(self, pos):
        row = pos[1]
        col = pos[1]
        ret = []
        for idx in range(col , row * self.size[1] , self.size[1]):
            if not self.cells[idx].blank:
                ret.append(self.cells[idx])
        for idx in range((row + 1) * self.size[1] + col , self.size[0] * self.size[1] , self.size[1]):
            if not self.cells[idx].blank:
                ret.append(self.cells[idx])
        return ret

    def get_context(self, pos):
        return self.get_row_context(pos) + self.get_col_context(pos)

    def get_entity_context(self, entity):
        ## entity context are pre-computed thus not computed here
        pass

##
# def load_table():
#     path = '/home/xusheng/TableProject/data/raw_table.txt'
#     tables = []
#     with open(path, 'r') as f:
#         data = json.load(f)
#         for tab in data:
#            table = Table(tab)
#            tables.append(table)
#     return tables
