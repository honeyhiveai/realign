

class Idea:
    def __init__(
        self, 
        seed: str, 
        depth: int = 0, 
        lineage: list["Idea"] = []
    ):
        self.seed = seed
        self.depth = depth
        self.lineage = lineage.copy()
        
        # Content
        self.content = ''
        
        # Ratings
        self.abs_rating = 0 # absolute rating
        self.elo_rating = 1000 # elo rating
    

    def __lt__(self, other: "Idea") -> bool:
        """
        Implements the less than comparison for Idea objects.
        This allows Idea instances to be compared and sorted in priority queues.
        
        Args:
            other (Idea): The other Idea object to compare with.
        
        Returns:
            bool: True if this Idea is considered less than the other, False otherwise.
        """
        return self.abs_rating < other.abs_rating

    def __eq__(self, other: "Idea") -> bool:
        """
        Implements the equality comparison for Idea objects.
        
        Args:
            other (Idea): The other Idea object to compare with.
        
        Returns:
            bool: True if the Ideas are considered equal, False otherwise.
        """
        return self.abs_rating == other.abs_rating
