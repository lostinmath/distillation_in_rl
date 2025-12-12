import logging
import mlflow
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class Logger:
    # TODO: Add normal logger here, I actually need those logs for plotting
    def __init__(
        self,
        run_name: str,
        use_mlflow=False,
        silence=True,
        log_file: str = "training_metrics_logs.txt",
        
    ) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.terminal_logger = logging.getLogger(__name__)
        self.use_mlflow = use_mlflow
        self.silence = silence  # basically a silance mode to disactivate logging
        self.log_file = log_file
        try:
            self.tensorboard_writer = SummaryWriter('tensorboard_logs/' + run_name + '_' + datetime.now().strftime('%D-%T'))
        except ImportError:
            self.tensorboard_writer = None
            self.terminal_logger.warning("TensorBoard is not installed. TensorBoard logging will be disabled.")
        # Ensure the log file exists

    def log_param(self, key, value):
        if self.use_mlflow:
            mlflow.log_param(key, value)
        else:
            if not self.silence:
                self.terminal_logger.info(f"{key}: {value}")
        
        # Log to TensorBoard
        if hasattr(self, 'tensorboard_writer'):
            self.tensorboard_writer.add_text(key, str(value))

    def log_metrics(self, metric_name, metric_value, step=-1):

        if True: #self.use_mlflow:
            mlflow.log_metrics({metric_name: metric_value}, step=step)
        else:
            if not self.silence:
                self.terminal_logger.info(f"{metric_name}: {metric_value}")
        # Log to TensorBoard
        if hasattr(self, 'tensorboard_writer') and self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar(metric_name, metric_value, global_step=step)

    def terminal_only_print(self, str_to_print):
        if not self.silence:
            self.terminal_logger.info(str_to_print)

    def log_to_file(self, str_to_log):
        with open(self.log_file, "a") as f:
            f.write(str_to_log + "\n")
        f.close()
