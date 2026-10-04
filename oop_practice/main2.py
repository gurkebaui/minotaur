class BaseBlock:
    def __init__(self, is_trained):
        self.is_trained = False
class ConvBlock(BaseBlock):
    def __init__(self, is_trained):
        super().__init__(is_trained)
        self.channels = is_trained

    def print(self):
        
        print("is trained", self.channels)

cb = ConvBlock(False)
cb.print()
