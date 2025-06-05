import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def clean_aircraft_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the aircraft data DataFrame by removing rows with NaN values and resetting the index.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing aircraft data.

    Returns:
    pd.DataFrame: The cleaned DataFrame with NaN values removed and index reset.
    """
    # Remove rows with NaN values
    df_cleaned = df.dropna()

    # Reset the index of the cleaned DataFrame
    df_cleaned.reset_index(drop=True, inplace=True)

    # Add Column for cruise speed in m/s from km/h
    df_cleaned['cruise_speed_mps'] = df_cleaned['cruise_speed_kmh'] * 1000 / 3600


    return df_cleaned

def regression_fit(x, y):
    X = np.array([x**3, x**2, x, np.ones_like(x)]).T
    print("X shape:", X.shape)
    U, sigma, Vh = np.linalg.svd(X, full_matrices=True)
    print("U shape:", U.shape)
    print("Sigma shape:", sigma.shape)
    print("Vh shape:", Vh.shape)
    print(Vh.T @ np.diag(1/sigma))
    return Vh.T @ np.hstack((np.diag((1/sigma)), np.zeros((4,17)))) @ U.T @ y

if __name__ == "__main__":
    df = pd.read_csv('aircraft_data.csv')  # Replace with your actual file path
    cleaned_df = clean_aircraft_data(df)
    df_8_14 = cleaned_df[(cleaned_df['wingspan_m'] >= 8) & (cleaned_df['wingspan_m'] <= 14)]
    df_15_20 = cleaned_df[(cleaned_df['wingspan_m'] >= 15) & (cleaned_df['wingspan_m'] <= 20)]
    df_21_30 = cleaned_df[(cleaned_df['wingspan_m'] >= 21) & (cleaned_df['wingspan_m'] <= 30)]
    df_31_40 = cleaned_df[(cleaned_df['wingspan_m'] >= 31)]

    beta = regression_fit(cleaned_df['wingspan_m'], cleaned_df['cruise_speed_mps'])
    t = np.linspace(0, 40, 100)
    cruise_speed_fit = beta[0] * t**3 + beta[1] * t**2 + beta[2] * t + beta[3]
    plt.scatter(cleaned_df['wingspan_m'], cleaned_df['cruise_speed_mps'], alpha=0.5)
    plt.plot(t, cruise_speed_fit, color='red', label='Regression Fit')
    plt.xlabel('Wingspan (m)')
    plt.ylabel('Cruise Speed (m/s)')
    plt.title('Wingspan vs Cruise Speed')
    plt.show()

    print(df_8_14)
    plt.hist(df_8_14['cruise_speed_mps'], bins=5, alpha=0.5, label='8-14 m wingspan')
    plt.show()

    print(df_15_20)
    plt.hist(df_15_20['cruise_speed_mps'], bins=5, alpha=0.5, label='15-20 m wingspan')
    plt.show()

    print(df_21_30)
    plt.hist(df_21_30['cruise_speed_mps'], bins=5, alpha=0.5, label='21-30 m wingspan')
    plt.show()

    print(df_31_40)
    plt.hist(df_31_40['cruise_speed_mps'], bins=5, alpha=0.5, label='31-40 m wingspan')
    plt.show()

    
