class Base:
    def __call__(self):
        raise NotImplementedError("Not implemented, call apply()")

    def apply(self):
        raise NotImplementedError

    def release(self):
        raise NotImplementedError
