class PokerHandEvaluator:
    RANKS = "23456789TJQKA"
    SUITS = "cdhs"
    FULL_NAMES = {
        'Two': '2',
        'Three': '3',
        'Four': '4',
        'Five': '5',
        'Six': '6',
        'Seven': '7',
        'Eight': '8',
        'Nine': '9',
        'Ten': 'T',
        'Jack': 'J',
        'Queen': 'Q',
        'King': 'K',
        'Ace': 'A'
    }
    CARD_VALUES_TO_NAMES = {
        2: 'Two',
        3: 'Three',
        4: 'Four',
        5: 'Five',
        6: 'Six',
        7: 'Seven',
        8: 'Eight',
        9: 'Nine',
        10: 'Ten',
        11: 'Jack',
        12: 'Queen',
        13: 'King',
        14: 'Ace'
    }
    SUIT_MAP = {
        'Spade': 's',
        'Heart': 'h',
        'Club': 'c',
        'Diamond': 'd'
    }

    def __init__(self, holding, board):
        # Ensure each input is split into individual card strings
        self.holding = self.split_concatenated_cards(holding)
        self.board = [card for part in board for card in self.split_concatenated_cards(part)]
        self.all_cards = self.holding + self.board
        # print(f"holding {self.holding}")
        # print(f"board {self.board}")

    def split_concatenated_cards(self, cards):
        """Ensures that cards are split into two-character or three-character strings as needed."""
        if isinstance(cards, list):
            return [card.strip('[]') for part in cards for card in self.split_concatenated_cards(part)]
        elif isinstance(cards, str):
            cleaned_cards = cards.replace("[", "").replace("]", "").strip()  # Remove brackets and spaces
            parsed_cards = []
            i = 0
            while i < len(cleaned_cards):
                # Check if the rank is 10 (three characters), otherwise two characters
                if cleaned_cards[i] == '1' and i + 2 < len(cleaned_cards) and cleaned_cards[i + 1] == '0':
                    parsed_cards.append(cleaned_cards[i:i + 3])  # Take three characters (e.g., '10s')
                    i += 3
                else:
                    parsed_cards.append(cleaned_cards[i:i + 2])  # Take two characters (e.g., 'Qs')
                    i += 2
            return parsed_cards
        else:
            raise ValueError("Cards should be provided as a list or a concatenated string.")
                
    def get_card_value(self, card):
        """Extracts the rank value (2-14) and suit from a card like 'Ts', '8c', or '10s'."""
        if isinstance(card, str):
            if len(card) == 3 and card[:2] == '10':
                rank_value = 10
                suit = card[2]
            elif len(card) == 2:  
                rank_char = card[0]
                if rank_char not in self.RANKS:
                    raise ValueError(f"Invalid rank character '{rank_char}' in card '{card}'")
                rank_value = self.RANKS.index(rank_char) + 2  # Map rank index to rank value (2-14)
                suit = card[1]
            else:
                raise ValueError(f"Invalid card format {card}")
            return rank_value, suit
        else:
            raise ValueError(f"Invalid card format {card}")

    def check_flush(self, cards):
        """Checks if there is a flush in the given cards and returns the highest cards if flush exists."""
        suit_count = {suit: [] for suit in self.SUITS}
        # print(cards)
        for card in cards:
            rank, suit = self.get_card_value(card)
            suit_count[suit].append(rank)

        for suit, ranks in suit_count.items():
            if len(ranks) >= 5:
                # Sort ranks in descending order to get the highest cards in the flush
                ranks.sort(reverse=True)
                return True, ranks[:5]  # Return top 5 cards of the flush
        return False, []

    def check_pairs(self, cards):
        """Identifies pairs, three of a kinds, and four of a kinds in the cards."""
        rank_count = {}
        for card in cards:
            rank_value, suit = self.get_card_value(card)
            rank_count[rank_value] = rank_count.get(rank_value, 0) + 1

        pairs = [rank for rank, count in rank_count.items() if count == 2]
        threes = [rank for rank, count in rank_count.items() if count == 3]
        fours = [rank for rank, count in rank_count.items() if count == 4]

        return pairs, threes, fours

    def parse_holding_string(self, holding_str):
        """Parses a descriptive holding string like 'King of Diamond and King of Club' to ['Kd', 'Kc']."""
        holding_str = holding_str.replace("[", "").replace("]", "").strip()
        
        cards = holding_str.split(" and ")
        parsed_holding = []
        for card in cards:
            rank_word, suit_word = card.split(" of ")
            rank = self.FULL_NAMES[rank_word.strip()]  # Strip any leading/trailing whitespace
            suit = self.SUIT_MAP[suit_word.strip()]  # Strip any leading/trailing whitespace
            parsed_holding.append(rank + suit)
        
        return parsed_holding

    def check_straight(self, cards):
        """Checks if there is a straight in the given cards and returns the highest cards if a straight exists."""
        if all(isinstance(card, int) for card in cards):
            unique_ranks = sorted(set(cards), reverse=True)
        else:
            unique_ranks = sorted(set(self.get_card_value(card)[0] for card in cards), reverse=True)
            
        if 14 in unique_ranks:
            unique_ranks.append(1)

        consecutive = []
        for rank in unique_ranks:
            if not consecutive or consecutive[-1] - rank == 1:
                consecutive.append(rank)
                if len(consecutive) == 5:
                    return True, consecutive  
            else:
                consecutive = [rank]

        return False, []


    def rank_to_name(self, rank_value, full_name=False):
        """Converts a rank value to its full name or abbreviation."""
        name = self.CARD_VALUES_TO_NAMES.get(rank_value)
        if name:
            if full_name:
                return name
            else:
                return self.FULL_NAMES[name]
        else:
            return None

    def get_best_hand(self):
        """Finds the best possible hand from the holding and the board."""
        if not self.board:
            rank1, suit1 = self.get_card_value(self.holding[0])
            rank2, suit2 = self.get_card_value(self.holding[1])

            if rank1 == rank2:
                return "One Pair", f"One Pair, {self.rank_to_name(rank1, full_name=True)}s"
            else:
                high_card_rank = max(rank1, rank2)
                return "High Card", f"{self.rank_to_name(high_card_rank, full_name=True)}-high"

        # Combine holding and board if the board is not empty
        all_cards = self.holding + self.board

        # Check for Straight Flush / Royal Flush
        flush, flush_cards = self.check_flush(all_cards)
        # print(f"is flush {flush, flush_cards}")
        if flush:
            straight_flush, straight_flush_cards = self.check_straight(flush_cards)
            # print(f"is straight flush {straight_flush, straight_flush_cards}")
            
            # Corrected check for Royal Flush
            if straight_flush and straight_flush_cards[0] == 14 and straight_flush_cards[-1] == 10:
                return "Royal Flush", "Ace-high Royal Flush"
            
            if straight_flush:
                return "Straight Flush", f"Straight Flush, {self.rank_to_name(straight_flush_cards[-1])} to {self.rank_to_name(straight_flush_cards[0])}"

        # Check for Four of a Kind
        pairs, threes, fours = self.check_pairs(all_cards)
        if fours:
            kicker = max([self.get_card_value(card)[0] for card in all_cards if self.get_card_value(card)[0] not in fours])
            return "Four of a Kind", f"Four of a Kind, {self.rank_to_name(fours[0], full_name=True)}s with {self.rank_to_name(kicker)} kicker"

        # Check for Full House
        if threes and pairs:
            return "Full House", f"Full House, {self.rank_to_name(threes[0], full_name=True)}s over {self.rank_to_name(pairs[0], full_name=True)}s"
        if len(threes) > 1:  # Three of a kind with another three
            return "Full House", f"Full House, {self.rank_to_name(threes[0], full_name=True)}s over {self.rank_to_name(threes[1], full_name=True)}s"

        # Check for Flush
        if flush:
            return "Flush", f"Flush, {self.rank_to_name(flush_cards[0], full_name=True)}-high"
        
        # Check for Straight
        straight, straight_cards = self.check_straight(all_cards)
        if straight:
            return "Straight", f"Straight, {self.rank_to_name(straight_cards[-1], full_name=True)} to {self.rank_to_name(straight_cards[0], full_name=True)}"

        # Check for Three of a Kind
        if threes:
            kicker_cards = sorted([self.get_card_value(card)[0] for card in all_cards if self.get_card_value(card)[0] not in threes], reverse=True)
            return "Three of a Kind", f"Three of a Kind, {self.rank_to_name(threes[0], full_name=True)}s with {self.rank_to_name(kicker_cards[0])} and {self.rank_to_name(kicker_cards[1])} kickers"

        # Check for Two Pair - Explicit ordering adjustment and kicker formatting
        if len(pairs) > 1:
            high_pair, low_pair = sorted(pairs, reverse=True)[:2]
            kicker = max([self.get_card_value(card)[0] for card in all_cards if self.get_card_value(card)[0] not in pairs])
            return "Two Pair", f"Two Pair, {self.rank_to_name(high_pair, full_name=True)}s and {self.rank_to_name(low_pair, full_name=True)}s with {self.rank_to_name(kicker, full_name=True)} kicker"

        # Check for One Pair
        if pairs:
            kicker_cards = sorted(
                [self.get_card_value(card)[0] for card in all_cards if self.get_card_value(card)[0] not in pairs],
                reverse=True
            )
            return "One Pair", (
                f"One Pair, {self.rank_to_name(pairs[0], full_name=True)}s with "
                f"{', '.join(self.rank_to_name(k, full_name=True) for k in kicker_cards[:3])} kickers"
            )

        # If nothing, return High Card
        all_ranks = sorted([self.get_card_value(card)[0] for card in all_cards], reverse=True)
        return "High Card", f"{self.rank_to_name(all_ranks[0])}-high"