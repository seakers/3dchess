class DealerPreplanner(Preplanner):
    """
    A preplanner that generates plans for other agents.
    """

    def __init__(self, replanner: Replanner, debug: bool = False, logger: Optional[Logger] = None):
        super().__init__(replanner, debug, logger)
        self._replanner = replanner

    def plan(self, state: State) -> None:
        """
        Plan the dealer's actions based on the current state.
        """
        self._replanner.plan(state)