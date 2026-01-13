#!/usr/bin/env python3
"""Simple CLI Coffee Machine simulator.

Commands:
 - `buy` to purchase a drink (espresso/latte/cappuccino)
 - `report` to show resources and earnings
 - `refill` to reset resources to defaults
 - `off` to exit

This is a self-contained script meant for local use.
"""
from typing import Dict

MENU = {
    "espresso": {"ingredients": {"water": 50, "coffee": 18}, "cost": 1.5},
    "latte": {"ingredients": {"water": 200, "milk": 150, "coffee": 24}, "cost": 2.5},
    "cappuccino": {"ingredients": {"water": 250, "milk": 100, "coffee": 24}, "cost": 3.0},
}

DEFAULT_RESOURCES = {"water": 300, "milk": 200, "coffee": 100}


class CoffeeMachine:
    def __init__(self):
        self.resources: Dict[str, int] = DEFAULT_RESOURCES.copy()
        self.money: float = 0.0

    def report(self) -> None:
        print("Resources:")
        for k, v in self.resources.items():
            unit = "ml" if k in ("water", "milk") else "g"
            print(f"  {k.capitalize()}: {v}{unit}")
        print(f"  Money: ${self.money:.2f}")

    def is_resource_sufficient(self, drink: str) -> bool:
        ingredients = MENU[drink]["ingredients"]
        for item, amount in ingredients.items():
            if self.resources.get(item, 0) < amount:
                print(f"Sorry, not enough {item}.")
                return False
        return True

    def process_coins(self) -> float:
        print("Please insert coins.")
        try:
            quarters = int(input("  how many quarters?: "))
            dimes = int(input("  how many dimes?: "))
            nickels = int(input("  how many nickels?: "))
            pennies = int(input("  how many pennies?: "))
        except ValueError:
            print("Invalid coin input. Transaction cancelled.")
            return 0.0
        total = quarters * 0.25 + dimes * 0.10 + nickels * 0.05 + pennies * 0.01
        return round(total, 2)

    def make_coffee(self, drink: str) -> None:
        ingredients = MENU[drink]["ingredients"]
        for item, amount in ingredients.items():
            self.resources[item] -= amount
        self.money += MENU[drink]["cost"]
        print(f"Here is your {drink}. Enjoy!")

    def refill(self) -> None:
        self.resources = DEFAULT_RESOURCES.copy()
        print("Resources refilled to defaults.")

    def run(self) -> None:
        print("Welcome to the Coffee Machine. Type 'help' for commands.")
        try:
            while True:
                cmd = input("Enter command (buy/report/refill/off/help): ").strip().lower()
                if cmd == "off":
                    print("Shutting down. Goodbye!")
                    break
                elif cmd == "report":
                    self.report()
                elif cmd == "refill":
                    self.refill()
                elif cmd == "help":
                    print("Commands: buy, report, refill, off")
                elif cmd == "buy":
                    choice = input("What would you like? (espresso/latte/cappuccino): ").strip().lower()
                    if choice not in MENU:
                        print("Unknown drink. Try again.")
                        continue
                    if not self.is_resource_sufficient(choice):
                        continue
                    payment = self.process_coins()
                    cost = MENU[choice]["cost"]
                    if payment < cost:
                        print("Sorry that's not enough money. Money refunded.")
                        continue
                    change = round(payment - cost, 2)
                    if change > 0:
                        print(f"Here is ${change:.2f} in change.")
                    self.make_coffee(choice)
                else:
                    print("Unknown command. Type 'help'.")
        except (KeyboardInterrupt, EOFError):
            print("\nShutting down. Goodbye!")


def main():
    machine = CoffeeMachine()
    machine.run()


if __name__ == "__main__":
    main()
