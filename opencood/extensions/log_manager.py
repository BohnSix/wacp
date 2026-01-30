import builtins
import os
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import colorama
import re

class LogManager:

    class ColoredFormatter(logging.Formatter):
        colorama.init()

        COLORS = {
            'DEBUG': colorama.Fore.BLUE,
            'INFO': colorama.Fore.GREEN,
            'WARNING': colorama.Fore.YELLOW,
            'ERROR': colorama.Fore.RED,
            'CRITICAL': colorama.Fore.MAGENTA,
            'PRINT': colorama.Fore.WHITE
        }

        def format(self, record):
            log_message = super().format(record)
            levelname = record.levelname
            if record.msg.startswith('[PRINT]'):
                levelname = 'PRINT'
            if levelname in self.COLORS:
                log_message = self.COLORS[levelname] + log_message + colorama.Style.RESET_ALL
            return log_message
        
    def __init__(self, log_dir='logs', verbose=True, colored_print=True, overide_print=True):
        """
        Initialize the LogManager.

        Args:
            log_dir (str): Directory to save log files. Default is 'logs'.
            verbose (bool): Whether to print log messages to the console. Default is True.
            colored_print (bool): Whether to use colored print for log messages in the console. Default is True.
            overide_print (bool): Whether to override the built-in print function. Default is True.
        Returns:
            None

        The LogManager class is responsible for managing logging, storing training and validation metrics,
        and visualizing them. It initializes the logging configuration, creates log files, and adjusts the log level.
        It also initializes lists to store training and validation metrics.
        """
        self.log_dir = log_dir
        self.verbose = verbose
        self.log_id = self.get_time_stamp()
        os.makedirs(log_dir, exist_ok=True)

        if overide_print is True:
            # Override the built - in print function
            self.original_print = builtins.print
            builtins.print = self.custom_print

        # Configure logging
        log_file_path = os.path.join(log_dir, f'training_{self.log_id}.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # File handler to remove ANSI escape sequences
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(lambda record: setattr(record, 'msg', self._remove_ansi_escape(record.msg)) or True)

        if verbose:
            # Console handler using custom formatter to keep colors
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            if colored_print is True:
                console_formatter = self.ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s')
                console_handler.setFormatter(console_formatter)
            handlers = [file_handler, console_handler]
        else:
            handlers = [file_handler]

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
        self.logger = logging.getLogger(__name__)

        # Adjust the log level of matplotlib.font_manager
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

        # Lists to store training and validation metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        # Log initialization information
        self.log_info(f'LogManager initialized. \n Log ID: {self.log_id}\n Log DIR: {os.path.abspath(log_file_path)}\n')

    def _remove_ansi_escape(self, text):
        """
        Remove ANSI escape sequences from a given text.

        This function is used to remove ANSI escape sequences from log messages,
        which are used to colorize log messages in the console.

        Parameters:
        text (str): The input text containing ANSI escape sequences.

        Returns:
        str: The input text with ANSI escape sequences removed.
        """
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    def custom_print(self, *args, sep=' ', end='\n'):
        """
        Custom print function that logs the printed messages to the logger.

        This function overrides the built-in print function to log the printed messages to the logger.
        It joins the provided arguments into a single string using the specified separator and end character.
        The resulting message is then logged using the log_info method.

        Parameters:
        *args: Variable length argument list. The arguments to be printed.
        sep (str): The separator between the arguments. Default is a single space.
        end (str): The string to be appended at the end of the printed message. Default is a newline character.

        Returns:
        None
        """
        message = sep.join(str(arg) for arg in args) + end
        self.log_info(f'[PRINT] {message.rstrip()}')

    def get_time_stamp(self):
        return datetime.now().strftime("%Y%m%d-%H%M%S")

    def log_info(self, message):
        """
        Log an info message to the logger and console.

        This function logs the provided message at the INFO level. It uses the
        configured logger to write the message to a log file and, if verbose mode
        is enabled, prints the message to the console.
        Args:
            message (str): The message to log. The message should be a string.

        Returns:
            None
        """
        self.logger.info(f'{message}')


    def log_error(self, message):
        """
        Log an error message to the logger and console.

        This function logs the provided error message at the ERROR level. It uses the
        configured logger to write the message to a log file and, if verbose mode
        is enabled, prints the message to the console.
        Args:
            message (str): The error message to log. The message should be a string.

        Returns:
            None
        """
        self.logger.error(f'{message}')


    def log_debug(self, message):
        """
        Log a debug message to the logger.

        This function logs the provided debug message at the DEBUG level. It uses the
        configured logger to write the message to a log file. The message will not be printed
        to the console unless verbose mode is enabled.
        Args:
            message (str): The debug message to log. The message should be a string.

        Returns:
            None
        """
        self.logger.debug(f'{message}')

    def save_metrics(self, train_loss, val_loss, train_accuracy, val_accuracy, info=None):
        """
        Save training and validation metrics.

        This function appends the provided training and validation metrics to their respective lists,
        and logs these metrics to the logger.
        Args:
            train_loss (float): The training loss for the current epoch.
            val_loss (float): The validation loss for the current epoch.
            train_accuracy (float): The training accuracy for the current epoch.
            val_accuracy (float): The validation accuracy for the current epoch.
            info (str, optional): Additional information to be logged. Defaults to None.

        Returns:
            None
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_accuracy)
        self.val_accuracies.append(val_accuracy)

        self.log_info(f'Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        self.log_info(f'Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')
        if info is not None:
            self.log_info(f'Extra info: {info}')


    def visualize_metrics(self):
        """
        Visualize training and validation metrics.

        This function creates a plot to visualize the training and validation metrics.
        The plot consists of two subplots: one for losses and one for accuracies.
        The losses are plotted against epochs, and the accuracies are plotted against epochs.
        The plot is saved as an image file in the log directory.

        Parameters:
        None

        Returns:
        None

        The function uses the following attributes:
        - self.train_losses: A list containing the training losses for each epoch.
        - self.val_losses: A list containing the validation losses for each epoch.
        - self.train_accuracies: A list containing the training accuracies for each epoch.
        - self.val_accuracies: A list containing the validation accuracies for each epoch.
        - self.log_dir: The directory where log files are saved.
        - self.log_id: The unique identifier for the current log session.
        """
        plt.figure(figsize=(12, 5))

        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Training Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plot_file_path = os.path.join(self.log_dir, f'metrics_plot_{self.log_id}.png')
        plt.savefig(plot_file_path)
        self.log_info(f'Metrics plot saved: {os.path.abspath(plot_file_path)}')

# Example usage
if __name__ == "__main__":
    log_manager = LogManager(log_dir='running_logs', verbose=True, colored_print=True, overide_print=True)

    # Simulate training
    num_epochs = 10
    for epoch in range(num_epochs):
        # Dummy training and validation metrics
        train_loss = 1.0 / (epoch + 1)
        val_loss = 1.1 / (epoch + 1)
        train_accuracy = 0.8 + 0.01 * epoch
        val_accuracy = 0.7 + 0.01 * epoch

        log_manager.save_metrics(train_loss, val_loss, train_accuracy, val_accuracy)
        log_manager.log_info(f'A info example')
        log_manager.log_info(f'Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        log_manager.log_debug(f'Debug info at epoch {epoch}')
        log_manager.log_error("An error example")

    print("A print example")
    log_manager.visualize_metrics()
