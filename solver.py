# Seahaven towers game solver

from enum import Enum
from typing import List
from hashlib import sha256
from functools import reduce
import numpy as np
import time
from rich.console import Console
from rich.table import Table
from rich.text import Text
import csv
import tensorflow.keras as keras
import tensorflow as tf

model = keras.models.load_model("solverModel.keras")

console = Console()

ABANDON_THRESHOLD = 75000

class CardRank(Enum):
	Ace = 1
	Deuce = 2
	Three = 3
	Four = 4
	Five = 5
	Six = 6
	Seven = 7
	Eight = 8
	Nine = 9
	Ten = 10
	Jack = 11
	Queen = 12
	King = 13

	def __str__(self):
		match self.value:
			case 1: return 'A'
			case 11: return 'J'
			case 12: return 'Q'
			case 13: return 'K'
			case _: return str(self.value)


class CardSuit(Enum):
	Club = 'C'
	Heart = 'H'
	Diamond = 'D'
	Spade = 'S'

	def __str__(self):
		return self.value

	def __int__(self):
		match self.value:
			case 'C': return 0
			case 'H': return 1
			case 'D': return 2
			case 'S': return 3


class Card(tuple):

	def __new__(cls, rank, suit):
		assert isinstance(rank, CardRank)
		assert isinstance(suit, CardSuit)
		return tuple.__new__(cls, (rank, suit))

	@property
	def rank(self):
		return self[0]

	@property
	def suit(self):
		return self[1]
	
	@property
	def color(self):
		match self.suit:
			case CardSuit.Club: return 'blue1'
			case CardSuit.Spade: return 'deep_sky_blue1'
			case CardSuit.Heart: return 'red3'
			case CardSuit.Diamond: return 'bright_red'

	def __str__(self):

		return str(self.rank)+str(self.suit)

	def __setattr__(self, *ignored):
		raise NotImplementedError

	def __delattr__(self, *ignored):
		raise NotImplementedError
	
	@classmethod
	def cardFromStr(cls,rankStr:str,suitStr:str):
		rank:CardRank = None
		match rankStr:
			case 'A': rank = CardRank.Ace
			case 'T': rank = CardRank.Ten
			case 'J': rank = CardRank.Jack
			case 'Q': rank = CardRank.Queen
			case 'K': rank = CardRank.King
			case _: rank = CardRank(int(rankStr))

		suit: CardSuit = CardSuit(suitStr)

		return cls.__new__(cls,rank,suit)

		



class Deck:
	def __init__(self):
		self.cards = [
			Card(rank, suit) for rank in CardRank for suit in CardSuit
		]
		

	def shuffle(self):
		np.random.shuffle(self.cards)

	def deal(self):
		return self.cards.pop()


class StackType(Enum):
	GOAL =1
	TABLEAU = 2
	CELL = 3


class Tally:

	def __init__(self):
		self.totalGames = 0
		self.winnable = 0
		self.losers = 0
		self.abandoned = 0
		self.correctGuess = 0


class Position: 

	def __init__(self,index,type: StackType):
		self.index = index
		self.type = type


class Move: 
	def __init__(self,source:Position,target:Position,extent):
		self.source = source
		self.target = target
		self.extent = extent



class Stack:

	def __init__(self):
		self.pile = []

	def lastCard(self) -> Card:
		if len(self.pile) > 0: return self.pile[-1]
		return None


# Used for hashing and sorting the deck
def cardValue(card: Card) -> int:

	if card is None:
		return 0
	return  int(card.suit) * 20 + card.rank.value

def stackVector(stack: Stack) -> int:
	if len(stack.pile) == 0: return [0]
	return list(map(lambda c: cardValue(c),stack.pile))

def stackValue(stack: Stack, bottom=False) -> int:	
	if len(stack.pile) == 0: return 0
	if bottom: return cardValue(stack.pile[0])
	else: return cardValue(stack.pile[-1])


def cardName(card: Card,fallback: str) -> str:
	if card is None: return fallback
	return str(card)

def cardColor(card: Card,fallback: str) -> str:
	if card is None: return fallback
	return card.color

def cardText(card: Card,defaultName = '-',defaultColor='white'):
	name = cardName(card,defaultName)
	style = cardColor(card,defaultColor)
	return Text(name,style=style)

def stackCardText(stack: Stack,index:int,defaultName = '-',defaultColor='white'):
	if index >= len(stack.pile): return cardText(None,defaultName,defaultColor)
	card = stack.pile[index]
	return cardText(card,defaultName,defaultColor)





class Board:
	# create a unique checksum for the boards current state. This is used to ensure we never repeat a configuration, as its
	# easy in this game to achieve the same configuration from multiple move possibilities
	# Goal configuration is not considered
	# Cells are sorted to ensure that any order of the same cards in the cells are considered to be the same configuration
	# Stacks are sorted by the bottom most card, again to remove consideration of order from the checksum
	@property
	def checksum(self) -> int:

		# get a sorted group of cells
		cells = sorted(map(lambda c: stackValue(c),self.cells)) # cells is now a storted array with int value for each card

		orderedStacks = sorted(self.stacks,key = lambda s: stackValue(s,bottom=True))
		stacks = []
		for tabStack in orderedStacks:
			stacks.append(list(map(cardValue,tabStack.pile)))

		# stacks is now an array of arrays

		hash = sha256()
		for stack in stacks:			
			hash.update(bytearray(stack))
		hash.update(bytearray(cells))
		return hash.hexdigest()

	# Initialize the board
	def __init__(self):
		
		self.clearBoard()

	def randomDeal(self):
		deck = Deck()
		deck.shuffle()

		self.clearBoard()

		# deal 5 cards each to the 10 stacks in the tableau
		for i in range(0,10):
			stack = self.stacks[i]
			for k in range(0,5):
				card = deck.deal()
				stack.pile.append(card)
			

		# deal last two cards into the cells
		self.cells[0].pile.append(deck.deal())
		self.cells[1].pile.append(deck.deal()) 

	@property
	def vector(self) -> List[int]:
		"""Return a vector representation of the board"""
		vec: List[int] = []
		for c in self.cells:
			vec = vec+stackVector(c)

		for s in self.stacks:
			vec = vec+stackVector(s)

		return vec






	# clear the board of cards and initialize the board components
	def clearBoard(self): 
		self.goals: List[Stack] = []
		self.cells: List[Stack] = []
		self.stacks: List[Stack] = []

		for i in range(0,4): self.goals.append(Stack()) 
		for i in range(0,4): self.cells.append(Stack()) 

		for _ in range(0,10):
			self.stacks.append(Stack())


	# load cards into the board
	def loadCards(self,cells:List[Card],goals:List[Card],tableaus:List[List[Card]]):
		self.clearBoard()
		for i,c in enumerate(cells):
			self.cells[i].pile.append(c)

		for tabIndex,stack in enumerate(tableaus):
			for c in stack:
				self.stacks[tabIndex].pile.append(c)

		# loading goals is going to be a little harder, since we need to fill in hidden items
		for i,c in enumerate(goals):
			for rankValue in range(1,c.rank.value+1):
				self.goals[i].pile.append(Card(CardRank(rankValue),c.suit))


		



	


	# you cannot create a sequence of more than 5 consecutive cards if a lower card of the same suit is higher in the stack.
	# Doing so will block that suit from ever making it to the goal, because you can only move 5 cards in sequence at once
	# e.g. with stack 2H 10H 9H 8H 7H 6H, moving the 5H on the end would cause a situation where the 2H could never be freed.
	# we can ensure this doesn't happen and reduce our possiblity tree
	def isBlockingMove(self,card:Card,targetStack:Stack,extentLength:int) -> bool:

		if len(targetStack.pile) + extentLength < 5:
			return False		

		foundLower = False
		
		count = self.stackOrderedCount(targetStack)	
		for stackCard in targetStack.pile:
			if stackCard.suit == card.suit and stackCard.rank.value < card.rank.value :
				foundLower = True
				break

		# if we found a lower card higher in the stack AND the counted sequence + extentLength ( how many cards we are moving onto the stack ) >= 5 , then its a blocking move, as it will
		# result in 6 or more cards in sequence with a lower card higher in the stack
		if foundLower and (count + extentLength) >= 5 :			
			return True
		

		return False
	

	# returns how many cards on the top of the stack are ordered ( inclusive ).  That is, there will always be at least one, unless the stack is empty
	def stackOrderedCount(self,stack:Stack) -> int:
		if len(stack.pile) == 0: return 0

		count = 1
		for i,stackCard in enumerate(reversed(stack.pile[1:])):
			pos = (len(stack.pile) - i) - 1
			nextCard = stack.pile[pos-1]
			if stackCard.suit == nextCard.suit and stackCard.rank.value == nextCard.rank.value-1:
				count += 1
			else :
				break

		return count


	# return a collection of all cell positions that have nothing in them
	def findFreeCells(self) -> List[Position] :
		freeCells: List[Position] = []
		for (stackIndex,stack) in enumerate(self.cells) :
			if len(stack.pile) == 0 :
				freeCells.append(Position(index = stackIndex,type = StackType.CELL))

		return freeCells
		
	

	# count how many free cells there are
	def freeCellCount(self) -> int:
		return len(self.findFreeCells())



	# an extent is a ordered set of cards ( starting with top most ) that is less or euqal to the number of freeCells+1
	# For example, the most basic extent is 1 card, and we don't need any free cells
	# we can move an extent of values 5,4,3 if there are 2 or more free cells
	# logic is simple:  move every card except the final one into the available free cells, move the final card to target, then move cards from cells back onto final card in new position
	# we will return the total number of cards in the extent, or 0 meaning there is no movable card
	def findExtent(self,stack: Stack) -> int:
		count = self.stackOrderedCount(stack)
			
		if count <= (self.freeCellCount()+1): return count 
		return 0

	# Success if 52 cards in the goal stacks
	def isSuccess(self) -> bool:
		goalCount = reduce(lambda x,y: x+y,map(lambda s: len(s.pile),self.goals))
		return goalCount == 52 # goal will have 52 cards if game is over
	


	# Check to see if the stack is fully ordered
	# a stack is considered to be fully ordered if any ordered sequence from the top of the stack down is made up of more than the available free cells + 1
	# ( once you've hit 6 cards, the only place you can move the top card is to the goal.  You'll fill up the available cells trying to move the whole sequence)
	def isFullyOrdered(self,stack:Stack) -> bool:
		if len(stack.pile) == 0: return True
			
		
		freeCells = self.freeCellCount()
		count = self.stackOrderedCount(stack);

		if count == len(stack.pile) and stack.pile[0].rank == CardRank.King: return True # that is, if the stack is entirely ordered, and the root is a king, there's no need to move it

		if not (len(stack.pile) > (freeCells + 1) ) : return False# impossible to be fully ordered unless stack size is greater than the available free cells + 1

		if count > (freeCells+1): return True

		return False


	# Resolve a position into a reference to a particlar card stack
	def resolvePosition(self,position:Position) -> Stack:
		
		stack = None

		match position.type :
			case StackType.GOAL : stack = self.goals[position.index]
			case StackType.CELL : stack = self.cells[position.index]
			case StackType.TABLEAU : stack = self.stacks[position.index]

		return stack
	
	# move the card at the top of one stack to the top of another stack
	def moveCard(self,source: Position,target:Position) -> None:
		sourceStack = self.resolvePosition(source)
		targetStack = self.resolvePosition(target)
		card = sourceStack.pile.pop()
		targetStack.pile.append(card)


	def isLegalMove(self,card:Card,target:Position,extentLength:int) -> bool:

		targetStack = self.resolvePosition(target)

		if target.type == StackType.GOAL:
			#  two conditions.  The card is an Ace, and the goal is empty
			#  -or- the target's card is the same suit, and exactly one less in card value
			if len(targetStack.pile) == 0: return (card.rank == CardRank.Ace)				
							
			# check if card value is same suit and exactly +1 in value
			targetCard = targetStack.pile[-1]
			return targetCard.suit == card.suit and targetCard.rank.value == (card.rank.value-1)
		

		if target.type == StackType.CELL :
			return len(targetStack.pile) == 0 # our only requiremnt if the target is a Cell is that the stack is empty

		# target must be stack, no need to check

		# empty tableau stack can only accept king
		if len(targetStack.pile) == 0 : return (card.rank == CardRank.King)

		# for all other TABLEAU moves, the top of the target stack must be same suit and one GREATER in value
		targetCard = targetStack.pile[-1]
		return targetCard.suit == card.suit and targetCard.rank.value == (card.rank.value+1) and not self.isBlockingMove(card = card, targetStack = targetStack, extentLength = extentLength)
	
	
	# there's no point in moving the root card of a tableau stack if all kings are in root position
	def isUselessMove(self,source:Position) -> bool:
		if source.type != StackType.TABLEAU : return False 

		stack = self.resolvePosition(source)

		if len(stack.pile) != 1 : return False 
		# count root kings.  
		rootKings = 0
		for stack in self.stacks :
			if len(stack.pile) > 0 and stack.pile[0].rank == CardRank.King: rootKings += 1			

		if rootKings < 4 : return False 
		return True

	# find legals moves from a source position
	# even though a card may have up to 3 legal moves, only one of them make sense to make in any given circumstance
	def findLegalMove(self,source:Position) -> Move:
		
		sourceStack = self.resolvePosition(source)

		if len(sourceStack.pile) > 0: # stack must have something in it
			card = sourceStack.pile[-1] # get the card at the top of the pile

			if self.isUselessMove(source): return None

			# first check, for each goal stack, if move to goal is a legal move
			for (stackIndex,_) in enumerate(self.goals):
				target = Position(stackIndex, type = StackType.GOAL)
				if self.isLegalMove(card,target, extentLength = 1) : return Move(source,target,extent =1)  

			# short-circuit here if source stack is fully ordered.  
			if source.type == StackType.TABLEAU and self.isFullyOrdered(sourceStack): return None # no reason to move fully ordered card except to goal ( see isFullyOrdered for full definition )

			extent = 0
			
			if source.type == StackType.TABLEAU:
				# stack to stack moves will use an extent
				extent = self.findExtent(stack = sourceStack)
				if extent > 0:
					card = sourceStack.pile[len(sourceStack.pile) - extent]
				else:
					return None #if we found no extent from a source that is a Tableau, it means there's nothing that can be moved from that stack

			# consider all moves that target the Tableau, and make sure we use the 'extent' card, not the top card
			for (i,_) in enumerate(self.stacks):
				target = Position (index = i, type = StackType.TABLEAU) 
				if target.index == source.index and source.type == StackType.TABLEAU: continue  # don't consider move if source and target are the same
				if self.isLegalMove(card,target, extentLength = extent) : return Move(source, target, extent) 
			

			# only thing left is targeting free cells
			if source.type == StackType.CELL : return None # a card in a cell should only move to a goal or stack, which have already been considered.  Short-circuit here if our card is in a cell
			# that is, don't move from cell to cell
			
			freeCells = self.findFreeCells()
			if len(freeCells) > 0 and extent <= 1 :
				return Move(source, target = freeCells[0], extent =1)

		return None


class MasterBoardMemory:

	def __init__(self) :
		self.boardSet: dict[int: bool]  = {}
		self.repeatsAvoided:int = 0

	def registerBoard(self,board:Board) -> bool:
		chk = board.checksum
		if chk in self.boardSet:
			self.repeatsAvoided += 1
			return True
		

		self.boardSet[chk] = True
			
		return False


	@property
	def size(self) -> int :
		return len(self.boardSet)
	


class Game:

	def __init__(self,tally: Tally,boardMemory: MasterBoardMemory,board:Board = None) :
		
		self.tally:Tally = tally
		self.boardMemory:MasterBoardMemory = boardMemory

		self.stackSize = 0
		self.totalMoves = 0
		self.gameMoves: List[Move] = []
		self.abandoned = False

		if board is None:
			self.board:Board = Board()
			self.board.randomDeal() # deal out the game
		else:
			self.board = board

	

	def recordMove(self,source:Position,target:Position,extent:int):
		self.gameMoves.append( Move(source, target, extent))
	

	def moveCard(self,source:Position,target:Position,extent:int) :
		self.recordMove(source,target, extent)
			
		self.board.moveCard(source,target)	
		self.totalMoves += 1

		if self.totalMoves % 5000 == 0 :
			self.display(title = "Playing") 
			# // Thread.sleep(forTimeInterval: 0.1)


	# We move an extent by moving extent-1 cards to free cells, moving the inner most card in the extent, then moving the remaining from the cells in reverse order
	# e.g. if we have an extent of values 5,4,3 moving to a target stack where top card is 6, move 3, 4 to free cells, move 5 -> target stack, then 4,3 to target stack in that order
	# this totals to (extent-1) * 2 + 1 total moves.  This amount should be used when undoing this action
	# assume there are enough free cells to do this
	def moveExtent(self,source:Position,target:Position,extent:int): 
			
		freeCells = self.board.findFreeCells()

		# the number of free cells must be at least the extent-1.  That is, we can move 1 card when theres no free cells, 2 if 1 free cell, etc.
		if len(freeCells) >= (extent - 1) :
			for i in range(0,extent-1):
				cellPosition = freeCells[i]
				self.moveCard(source,target = cellPosition,extent = extent)
			
			self.moveCard(source,target,extent)
			for i in reversed(range(0,extent-1)):
				cellPosition = freeCells[i]
				self.moveCard(source = cellPosition,target = target,extent = extent)



	
	def undoLastMove(self):
		if len(self.gameMoves) > 0 :
			gameMove = self.gameMoves.pop() # pull off the last move

			self.board.moveCard(source = gameMove.target, target = gameMove.source) # you can see that's reversed

	# Make the given move and recursively continue playing from the new configuration.
	# That is, we will make that move, then follow that line of the possibility tree recursively.  Otherwise, we fail out of the function
	def moveAndPlayOn(self,move:Move ) -> bool:

		# for TABLEAU -> TABLEAU, use move extent
		if move.extent > 1 and move.source.type == StackType.TABLEAU and move.target.type == StackType.TABLEAU :
			self.moveExtent(move.source,move.target,move.extent)
		else:
			self.moveCard(move.source, move.target, move.extent)
		
		# we made our move, now lets do some checks before we move on

		if self.board.isSuccess() : 
			return True # check for success
		
		repeatBoard = self.boardMemory.registerBoard(self.board)

		if not repeatBoard :  # don't continue unless move wasn't a repeat ( classic example of too many negatives:  continue if not repeated)
			success = self.cycleThroughCards() # recursively attempt to solve the new board configuration
			if success : return True # the path from this configuration succeeded, so return True		

		# at this point, we know that this configuration wasn't a success, it might be a repeat, or its attempt to solve from the new configuration resulted in failure
		# in either case, we undo the move we just made
		
		if move.extent > 1 :
			totalExtentMoves = (move.extent-1)*2 + 1  # each extent move is recorded as individual moves, so we need to back them all out individually
			
			for _ in range(0,totalExtentMoves):	self.undoLastMove() 
				
		else:			
			self.undoLastMove() 
		

		return False # return the fact that this did not succeed

	# our fundamental game loop.  Iterate over every Tableau and Cell stack, finding each legal move in the current configuration
	# then make that move.  This function will be called recursively from the moveAndPlanOn() to attempt to win from the new configuration
	def cycleThroughCards(self) -> bool:
		self.stackSize += 1

		success = False

		allBoardMoves : List[Move] = []

		if self.totalMoves > ABANDON_THRESHOLD :
			self.abandoned = True
			return True
		


		# iterate through all tableau stacks and cells, coalating the legal moves into allBoardMoves
		for stackIndex in range(0,14): # we will resolve this index as 10 Tableau stacks and 4 cells
			source: Position

			# determine the source position
			if stackIndex > 3 :
				source = Position(index = stackIndex-4, type = StackType.TABLEAU)
			else:
				source = Position(index = stackIndex, type = StackType.CELL) 			

			move = self.board.findLegalMove(source) 
			if move != None: allBoardMoves.append(move)


		allBoardMoves = sorted(allBoardMoves,key = lambda m: m.target.type.value)

		# play every move recorded for this configuration.
		for lm in allBoardMoves :			
			success = self.moveAndPlayOn(move = lm)
			if success:  break 		

		self.stackSize -= 1
		return success
	

	def replayGame(self) :
		# rewind the entire game based on the move stack
		moveCopy = self.gameMoves.copy()
		
		# undo all moves
		for m in moveCopy:
			self.undoLastMove()
			self.display(title = "Rewinding        ")
			
		
		self.display(title="Original Layout")
		time.sleep(5)
		for m in moveCopy:
			self.moveCard(m.source,m.target,m.extent)
			if m.extent <= 1 :
				self.display(title = "Replay     ") 
				time.sleep(0.05)


	
	def display(self,title: str) :


		
		console.clear()

		goalTable = Table(show_header=False)
		for _ in range(4): goalTable.add_column(width=3)

		rowTexts = list(map(lambda s:cardText(s.lastCard()),self.board.goals))
		goalTable.add_row(*rowTexts)

		cellTable = Table(show_header=False)
		for _ in range(4): cellTable.add_column(width=3)

		rowTexts = list(map(lambda s:cardText(s.lastCard()),self.board.cells))
		cellTable.add_row(*rowTexts)

		tableauTable = Table(show_header=False)
		maxLength = max(list(map(lambda s: len(s.pile),self.board.stacks)))

		for i in range(10): tableauTable.add_column(width=3)
		for i in range(20):
			rowTexts = list(map(lambda s:stackCardText(s,i,defaultName=''),self.board.stacks))
			tableauTable.add_row(*rowTexts)


		table = Table(title=title)
		topTable = Table(show_header=True)
		topTable.add_column('Goals')
		topTable.add_column('Cells')
		topTable.add_row(goalTable,cellTable)
		table.add_row(topTable)
		table.add_row(tableauTable)
		console.print(table)
	
		table = Table(title='Stats')
		table.add_column('Games Played')
		table.add_column('Winnable')
		table.add_column('Losers')
		table.add_column('Abandoned')
		table.add_column('Correct Guess')
		table.add_column('Stack Size')
		table.add_column('Total Moves')
		table.add_column('Unique Configurations')
		table.add_column('Repeats Avoided')
		table.add_row(str(self.tally.totalGames),
				str(self.tally.winnable),
				str(self.tally.losers),
				str(self.tally.abandoned),
				str(self.tally.correctGuess),
				str(self.stackSize),
				str(self.totalMoves),
				str(self.boardMemory.size),
				str(self.boardMemory.repeatsAvoided) )

		console.print(table)		
	
	



def main() :
	
	tally = Tally()

	with open('games.csv','a') as csvFile:

		csvWriter = csv.writer(csvFile)
		

		for i in range(0,10000):
			boardMemory = MasterBoardMemory()
			
			game = Game(tally,boardMemory)
			game.display(title = "Start")		

			boardVector = game.board.vector
			tensor = np.array([boardVector])
			tensor.transpose()
			predictions = model.predict(tensor, steps=1,verbose=1)
			guess = np.round(predictions[0][0])
			print("Predict:",guess)
			

			success = game.cycleThroughCards()

			tally.totalGames += 1
			gameResult = 0
			if (game.abandoned) : 
				gameResult = 3
				tally.abandoned += 1 
			else:
				if (success):  
					gameResult = 1
					tally.winnable += 1 
				else:  
					gameResult = 2
					tally.losers += 1 

			if gameResult == guess:
				tally.correctGuess += 1


			boardVector.append(gameResult)
			csvWriter.writerow(boardVector)			
			csvFile.flush()

			game.display(title = "Finished")
			# time.sleep(0.5)

			# if success : game.replayGame()


		


def playGameWithBoard(board:Board):

	tally = Tally()

	boardMemory = MasterBoardMemory()
	
	game = Game(tally,boardMemory,board=board)
	game.display(title = "Start")
	time.sleep(5)

	success = game.cycleThroughCards()

	tally.totalGames += 1
	if (game.abandoned) : tally.abandoned += 1 
	else:
		if (success):  tally.winnable += 1 
		else:  tally.losers += 1 

	

	game.display(title = "Finished")
	time.sleep(0.5)

	if success : game.replayGame()




main()