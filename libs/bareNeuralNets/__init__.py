import warnings

def warning_formatter(msg, category, filename, lineno,   line=None):
        return 'Custom formatting'
      
warnings.warn('This is a test Neural Network library using only numpy and bare minimum'
              'functionality for educational purposes.', Warning, stacklevel=3)
