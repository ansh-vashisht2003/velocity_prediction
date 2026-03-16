import pandas as pd
import matplotlib.pyplot as plt

# load dataset
df = pd.read_csv("data/fake_gas_gun_dataset.csv")

# remove spaces from column names (safety)
df.columns = df.columns.str.strip()


# 1️⃣ Powder Mass vs Velocity
def powder_vs_velocity():

    plt.figure()

    plt.scatter(df["powder mass"], df["Actual Velocity"])

    plt.xlabel("Powder Mass")
    plt.ylabel("Velocity")
    plt.title("Powder Mass vs Velocity")

    plt.grid(True)

    plt.show()


# 2️⃣ Projectile Mass vs Velocity
def projectile_mass_vs_velocity():

    plt.figure()

    plt.scatter(df["projectile mass"], df["Actual Velocity"])

    plt.xlabel("Projectile Mass")
    plt.ylabel("Velocity")
    plt.title("Projectile Mass vs Velocity")

    plt.grid(True)

    plt.show()


# 3️⃣ Density vs Momentum
def density_vs_momentum():

    momentum = df["projectile mass"] * df["Actual Velocity"]

    plt.figure()

    plt.scatter(df["Density"], momentum)

    plt.xlabel("Density")
    plt.ylabel("Momentum")

    plt.title("Density vs Momentum")

    plt.grid(True)

    plt.show()


# 4️⃣ Expected Velocity vs Actual Velocity
def expected_vs_actual():

    plt.figure()

    plt.scatter(df["Actual Velocity"], df["expected velocity"])

    plt.xlabel("Actual Velocity")

    plt.ylabel("Expected Velocity")

    plt.title("Expected Velocity vs Actual Velocity")

    plt.grid(True)

    plt.show()


# 5️⃣ Energy vs Projectile Mass
def energy_vs_mass():

    energy = 0.5 * df["projectile mass"] * df["Actual Velocity"] ** 2

    plt.figure()

    plt.scatter(df["projectile mass"], energy)

    plt.xlabel("Projectile Mass")

    plt.ylabel("Energy")

    plt.title("Energy vs Projectile Mass")

    plt.grid(True)

    plt.show()