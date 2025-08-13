"""
The most general class for rl methods, basically a wrapper,
so the pipeline calls these methods regardless of the model type
"""


class AbstractedRrAlgo:
    def __init__(self) -> None:
        pass

    def model_setup(self):
        """ "
        This method is called before training. E.g. Can set up random values or
        do some necessary work with memory.
        By default does nothing.
        """
        pass

    def start_training(self, num_steps=10000):
        """
        Train the model.
        Raises:
            NotImplementedError: If the method is not implemented for the model.
        """
        raise NotImplementedError("Method train for implemented for the model.")

    def model_post_setup(self):
        """
        Perform any necessary setup or initialization tasks after the model has been created.
        This method is intended to be overridden by subclasses to implement any specific
        post-setup logic required for the model. By default, this method does nothing.
        Example usage:
            class MyModel(RLModelParent):
                # Custom setup logic here
        """
        pass

    def visualize(self):
        """
        Visualizes the model's performance or data.
        This method should be implemented to provide a visual representation
        of the model's performance, data, or any other relevant information
        that aids in understanding the model's behavior and results.
        Raises:
            NotImplementedError: If the method is not implemented.
        """
        pass
        # raise NotImplementedError("Method visualize not implemented for the model.")

    def save(self, save_path: str):
        """
        Save the model to the specified path.
        Args:
            save_path (str): The path where the model should be saved.
        Raises:
            NotImplementedError: If the method is not implemented for the model.
        """
        pass

        # raise NotImplementedError("Method save for implemented for the model.")
        
    @classmethod
    def load(path:str):
        """
        Load the model from the specified path.
        Args:
            path (str): The path from which to load the model.
        Returns:
            The loaded model.
        Raises:
            NotImplementedError: If the method is not implemented for the model.
        """
        raise NotImplementedError("Method load for implemented for the model.")
        
    def to(self, device):
        """
        Move the model to the specified device.
        Args:
            device (str): The device to which the model should be moved.
        """
        raise NotImplementedError("Method to for implemented for the model.")
    
    def train(self):
        """
        Set to train mode
        """
        raise NotImplementedError("Method train for implemented for the model.")
    
    def eval(self):
        """
        Set to eval mode
        """
        raise NotImplementedError("Method eval for implemented for the model.")
    
    
        