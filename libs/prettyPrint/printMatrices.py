import numpy as np
import inspect
import curses


class PrettyPrint:
  
  def get_cursor_position(self):
    """Gets the current cursor position in the terminal.

    Returns:
      A tuple (y, x) representing the cursor position (row, column).
    """
    stdscr = curses.initscr()  # Initialize curses
    curses.curs_set(0)        # Make cursor invisible (optional)
    y, x = stdscr.getyx()     # Get cursor position

    # curses.endwin()           # De-initialize curses

    return y, x
  
  # Move cursor to row (y), column (x) (top-left corner is 0, 0)  
  def goto_xy(self, x, y):
    print(f"\033[{y};{x}H", end="")  # f-string for code readability

  def move_cursor(self, x, y):
    if (y < 0):
      # Move up (NOTE: y is inverted!!)
      print(f"\033[{abs(y)}A", end="")
    elif (y > 0):
      # Move down
      print(f"\033[{y}B", end="")
    if (x > 0):
      # Move right
      print(f"\033[{x}C", end="")
    elif (x < 0):
      # Move left
      print(f"\033[{abs(x)}D", end="")
      
  def get_matrix_name(self):
    frame = inspect.currentframe()
    frame = inspect.getouterframes(frame)[2]
    string = inspect.getframeinfo(frame[0]).code_context[0].strip()
    args = string[string.find('(') + 1:-1].split(',')
    
    names = []
    for i in args:
        if i.find('=') != -1:
            names.append(i.split('=')[1].strip())
        
        else:
            names.append(i)
    return names[0]
      
  @staticmethod
  def matrix(matrix, print_name=False):    
    if print_name:
      matrix_name = PrettyPrint().get_matrix_name()
      print(f'{matrix_name}: ')
    
    if len(np.shape(matrix)) < 2:
      matrix = np.array([matrix])
    
    rows, columns = np.shape(matrix)

    upper_boundary = '_'*2
    lower_boundary = u'\N{OVERLINE}'*2
    boundary_length = len(upper_boundary)
    data_length = 10
    boundary_padding = columns*data_length - boundary_length - 1

    print('', f"{upper_boundary:<{boundary_padding}}", upper_boundary)
    
    for r in range(rows):
      print('|', end='')
      for c in range(columns):
        print(f"{'%.4f' % matrix[r][c]:^{data_length}}", end='')
      print('|')
      
    print('', f"{lower_boundary:<{boundary_padding}}", lower_boundary)
