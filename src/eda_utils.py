import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

    # -------------------------
    # 1. Data Summarization 
    # -------------------------
    
def convert_to_datetime(df, columns=None, infer=True, errors="coerce"):
    """
    Convert specified columns (or inferred ones) to datetime type.

    Parameters:
    - df: pandas DataFrame
    - columns: list of columns to convert (if None, auto-detect object columns)
    - infer: whether to try inferring date-like columns automatically
    - errors: 'coerce' (invalid parsing becomes NaT) or 'raise'

    Returns:
    - DataFrame with converted datetime columns
    """

    df = df.copy()

    # If no columns provided, try object columns
    if columns is None:
        columns = df.select_dtypes(include=["object"]).columns.tolist()

    converted_cols = []

    for col in columns:
        try:
            # Try conversion
            converted = pd.to_datetime(df[col], errors=errors)

            # Only keep if conversion makes sense (at least some valid dates)
            if converted.notna().sum() > 0:
                df[col] = converted
                converted_cols.append(col)

        except Exception:
            continue

    return df, converted_cols

def get_datetime_columns(df):
    """
    Return already datetime columns in the DataFrame.
    """
    return df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()


def get_numerical_columns(df):
    """
    Identify numerical columns in a DataFrame.
    """
    numerical_cols = df.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    return numerical_cols

def get_categorical_columns(df):
    """
    Identify categorical columns in a DataFrame.
    """
    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    return categorical_cols


def get_boolean_columns(df):
    """
    Identify boolean columns in a DataFrame.
    """
    boolean_cols = df.select_dtypes(
        include=["bool"]
    ).columns.tolist()

    return boolean_cols


def get_date_columns(df):
    """
    Identify date/datetime columns in a DataFrame.
    """
    date_cols = df.select_dtypes(
        include=["datetime64[ns]"]
    ).columns.tolist()

    return date_cols

def identify_column_types(df):
    """
    Return all column types in a dictionary.
    """
    column_types = {
        "numerical_columns": get_numerical_columns(df),
        "categorical_columns": get_categorical_columns(df),
        "boolean_columns": get_boolean_columns(df),
        "date_columns": get_date_columns(df),
    }

    return column_types

def descriptive_statistics(df):
    """
    Generate descriptive statistics for numerical features.
    Returns a clean summary DataFrame.
    """

    num_cols = get_numerical_columns(df)

    if not num_cols:
        raise ValueError("No numerical columns found in the dataset.")

    desc = df[num_cols].describe().T  # transpose for readability

    # Additional useful statistics
    desc["median"] = df[num_cols].median()
    desc["variance"] = df[num_cols].var()
    desc["skewness"] = df[num_cols].skew()
    desc["kurtosis"] = df[num_cols].kurtosis()

    # Reorder columns nicely
    desc = desc[
        ["count", "mean", "median", "std", "min", "25%", "50%", "75%", "max", "variance", "skewness", "kurtosis"]
    ]

    return desc

# -------------------------
# 2. Data Quality Assessment 
# ------------------------- 

def missing_value_summary(df):
    """
    Returns a DataFrame summarizing missing values per column.
    """

    summary = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percent": (df.isnull().sum() / len(df)) * 100,
        "dtype": df.dtypes
    })

    summary = summary.sort_values(by="missing_percent", ascending=False)

    return summary


def suggest_missing_value_strategy(df, threshold_high=50, threshold_low=5):
    """
    Suggest handling strategy based on missing value percentage.

    Rules:
    - > threshold_high% → drop column
    - threshold_low–threshold_high% → impute
    - < threshold_low% → drop rows or simple imputation
    """

    summary = missing_value_summary(df)

    strategies = []

    for col in summary.index:
        pct = summary.loc[col, "missing_percent"]

        if pct == 0:
            strategy = "No action needed"
        elif pct > threshold_high:
            strategy = "Drop column (too many missing values)"
        elif pct > threshold_low:
            if df[col].dtype in ["object"]:
                strategy = "Impute with mode or 'Unknown'"
            elif "datetime" in str(df[col].dtype):
                strategy = "Impute with forward fill or median date"
            else:
                strategy = "Impute with median/mean"
        else:
            strategy = "Drop rows or light imputation"

        strategies.append(strategy)

    summary["strategy"] = strategies

    return summary


def missing_value_report(df):
    """
    Full report: missing values + recommended strategy.
    """
    report = suggest_missing_value_strategy(df)

    return report

# -------------------------
# 3. Univariate Analysis 
# ------------------------- 

def plot_histograms(df, bins=30, figsize=(12, 8)):
    """
    Plot histograms for all numerical columns in the DataFrame.

    Parameters:
    - df: pandas DataFrame
    - bins: number of histogram bins
    - figsize: size of the plot
    """

    num_cols = get_numerical_columns(df)

    if not num_cols:
        print("No numerical columns found.")
        return

    df[num_cols].hist(bins=bins, figsize=figsize, edgecolor="black")

    plt.suptitle("Histograms of Numerical Features", fontsize=16)
    plt.tight_layout()
    plt.show()

# -------------------------
# 4. Bivariate / Multivariate Analysis  
# ------------------------- 

def clean_data(df):
    """
    Ensure required columns exist and drop missing values.
    """
    required_cols = ["totalpremium", "totalclaims", "zipcode"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df.dropna(subset=required_cols)


def plot_scatter_by_zipcode(df, sample_top_n=5):
    """
    Scatter plot of TotalPremium vs TotalClaims grouped by ZipCode.
    Only top N ZipCodes by frequency are plotted for readability.
    """

    df = clean_data(df)

    # Select most common ZipCodes
    top_zips = df["zipcode"].value_counts().head(sample_top_n).index
    filtered_df = df[df["zipcode"].isin(top_zips)]

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=filtered_df,
        x="totalpremium",
        y="totalclaims",
        hue="zipcode",
        alpha=0.7
    )

    plt.title("TotalPremium vs TotalClaims by ZipCode")
    plt.xlabel("Total Premium")
    plt.ylabel("Total Claims")
    plt.legend(title="ZipCode")
    plt.tight_layout()
    plt.show()


def correlation_by_zipcode(df):
    """
    Compute correlation between TotalPremium and TotalClaims per ZipCode.
    """

    df = clean_data(df)

    correlations = (
        df.groupby("zipcode")[["totalpremium", "totalclaims"]]
        .corr()
        .unstack()
        .iloc[:, 1]
        .reset_index()
    )

    correlations.columns = ["zipcode", "Correlation"]

    return correlations.sort_values(by="Correlation", ascending=False)


def plot_correlation_heatmap(df):
    """
    Correlation matrix heatmap for numerical features.
    """

    df = clean_data(df)

    corr_matrix = df[["totalpremium", "totalclaims"]].corr()

    plt.figure(figsize=(5, 4))

    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)

    plt.title("Correlation Matrix: TotalPremium vs TotalClaims")
    plt.tight_layout()
    plt.show()

# -------------------------
# 5. Geographic Trends  
# ------------------------- 

def validate_columns(df):
    """
    Check required columns exist.
    """
    required_cols = ["province", "covertype", "totalpremium", "automake"]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")


def premium_by_province(df):
    """
    Average premium across provinces.
    """
    validate_columns(df)

    premium_summary = (
        df.groupby("province")["totalpremium"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    return premium_summary


def cover_type_distribution(df):
    """
    Distribution of cover types across provinces.
    """
    validate_columns(df)

    cover_summary = pd.crosstab(
        df["province"],
        df["covertype"]
    )

    return cover_summary


def auto_make_distribution(df):
    """
    Distribution of AutoMake across provinces.
    """
    validate_columns(df)

    auto_summary = pd.crosstab(
        df["province"],
        df["automake"]
    )

    return auto_summary


def plot_average_premium(df):
    """
    Bar chart of average premium by province.
    """
    premium_summary = premium_by_province(df)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=premium_summary,
        x="province",
        y="totalpremium"
    )

    plt.title("Average Total Premium by Province")
    plt.xlabel("Province")
    plt.ylabel("Average Premium")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_cover_type_heatmap(df):
    """
    Heatmap of cover type distribution by province.
    """
    cover_summary = cover_type_distribution(df)

    plt.figure(figsize=(12, 6))

    sns.heatmap(
        cover_summary,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Cover Type Distribution Across Provinces")
    plt.xlabel("Cover Type")
    plt.ylabel("Province")

    plt.tight_layout()
    plt.show()


def plot_top_auto_makes(df, top_n=10):
    """
    Plot most common auto makes across provinces.
    """
    validate_columns(df)

    top_makes = df["automake"].value_counts().head(top_n).index

    filtered_df = df[df["automake"].isin(top_makes)]

    plt.figure(figsize=(12, 6))

    sns.countplot(
        data=filtered_df,
        x="automake",
        hue="province"
    )

    plt.title("Top Auto Makes Across Provinces")
    plt.xlabel("Auto Make")
    plt.ylabel("Count")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# -------------------------
# 6. Outlier Detection   
# ------------------------- 

def plot_boxplots(df, columns=None, figsize=(12, 6)):
    """
    Plot boxplots for selected numerical columns.

    Parameters:
    - df: pandas DataFrame
    - columns: list of numerical columns (optional)
    - figsize: figure size
    """

    # Automatically detect numerical columns
    if columns is None:
        columns = get_numerical_columns(df)

    if not columns:
        print("No numerical columns found.")
        return

    # Create boxplots
    plt.figure(figsize=figsize)

    sns.boxplot(data=df[columns])

    plt.title("Box Plots of Numerical Features")
    plt.xlabel("Features")
    plt.ylabel("Values")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def plot_individual_boxplots(df, columns=None):
     """
     Plot boxplots for all numerical columns
      in a grid layout like the histogram image.
    """

     # Automatically get numerical columns
     if columns is None:
        columns = get_numerical_columns(df)

     if len(columns) == 0:
        print("No numerical columns found.")
        return

    # Number of columns in subplot grid
     n_cols = 3

    # Calculate rows
     n_rows = (len(columns) + n_cols - 1) // n_cols

    # Create figure
     fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(18, 5 * n_rows)
    )

    # Flatten axes
     axes = axes.flatten()

    # Plot each feature
     for i, col in enumerate(columns):

        sns.boxplot(
            y=df[col],
            ax=axes[i]
        )

        axes[i].set_title(col)
        axes[i].set_ylabel("")
        axes[i].grid(True)

    # Remove extra empty plots
     for j in range(len(columns), len(axes)):
        fig.delaxes(axes[j])

    # Main title
     fig.suptitle(
        "Box Plots of Numerical Features",
        fontsize=18
    )

     plt.tight_layout()

    # Prevent title overlap
     plt.subplots_adjust(top=0.92)

     plt.show()

# =========================================================
# % OVERALL LOSS RATIO ANALYSIS
# =========================================================

def calculate_loss_ratio(df):
    """
    Calculate overall portfolio loss ratio.
    Loss Ratio = TotalClaims / TotalPremium
    """

    total_claims = df["totalclaims"].sum()
    total_premium = df["totalpremium"].sum()

    loss_ratio = total_claims / total_premium

    print(f"Overall Loss Ratio: {loss_ratio:.2f}")

    return loss_ratio


def loss_ratio_by_group(df, group_col):
    """
    Calculate loss ratio grouped by a categorical feature.
    """

    grouped = (
        df.groupby(group_col)[["totalclaims", "totalpremium"]]
        .sum()
    )

    grouped["LossRatio"] = (
        grouped["totalclaims"] / grouped["totalpremium"]
    )

    return grouped.sort_values(by="LossRatio", ascending=False)


def plot_loss_ratio(df, group_col):
    """
    Plot loss ratio by category.
    """

    result = loss_ratio_by_group(df, group_col)

    plt.figure(figsize=(10, 5))

    sns.barplot(
        x=result.index,
        y=result["LossRatio"]
    )

    plt.title(f"Loss Ratio by {group_col}")
    plt.ylabel("Loss Ratio")
    plt.xlabel(group_col)

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

# =========================================================
# % DISTRIBUTION OF FINANCIAL VARIABLES
# =========================================================

def plot_financial_distributions(df):
    """
    Plot distributions for financial variables.
    """

    financial_cols = [
        "totalclaims",
        "customvalueestimate",
        "totalpremium"
    ]

    df[financial_cols].hist(
        figsize=(15, 5),
        bins=30,
        edgecolor="black"
    )

    plt.suptitle("Financial Variable Distributions")
    plt.tight_layout()
    plt.show()


def detect_outliers_iqr(df, column):
    """
    Detect outliers using IQR method.
    """

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    print(f"{column} Outliers: {len(outliers)}")

    return outliers

# =========================================================
# % TEMPORAL TREND ANALYSIS
# =========================================================

def prepare_date_column(df, date_col):
    """
    Convert date column to datetime.
    """

    df[date_col] = pd.to_datetime(df[date_col])

    return df


def monthly_claim_trends(df, date_col):
    """
    Analyze monthly claim frequency and severity.
    """

    df = prepare_date_column(df, date_col)

    df["YearMonth"] = df[date_col].dt.to_period("M")

    trends = (
        df.groupby("YearMonth")
        .agg(
            ClaimFrequency=("totalclaims", "count"),
            ClaimSeverity=("totalclaims", "mean")
        )
        .reset_index()
    )

    return trends


def plot_claim_trends(df, date_col):
    """
    Plot claim frequency and severity trends.
    """

    trends = monthly_claim_trends(df, date_col)

    trends["YearMonth"] = trends["YearMonth"].astype(str)

    plt.figure(figsize=(14, 5))

    plt.plot(
        trends["YearMonth"],
        trends["ClaimFrequency"],
        marker="o",
        label="Claim Frequency"
    )

    plt.plot(
        trends["YearMonth"],
        trends["ClaimSeverity"],
        marker="o",
        label="Claim Severity"
    )

    plt.title("Monthly Claim Trends")
    plt.xlabel("Month")
    plt.ylabel("Value")

    plt.xticks(rotation=45)

    plt.legend()

    plt.tight_layout()
    plt.show()

# =========================================================
# % VEHICLE MAKE / MODEL CLAIM ANALYSIS
# =========================================================

def vehicle_claim_analysis(df, vehicle_col="automake"):
    """
    Analyze claim amounts by vehicle make/model.
    """

    result = (
        df.groupby(vehicle_col)["totalclaims"]
        .mean()
        .sort_values(ascending=False)
    )

    highest = result.head(10)
    lowest = result.tail(10)

    print("\nHighest Claim Vehicles")
    print(highest)

    print("\nLowest Claim Vehicles")
    print(lowest)

    return result


def plot_vehicle_claims(df, vehicle_col="automake", top_n=10):
    """
    Plot top vehicle makes/models by average claim amount.
    """

    result = (
        df.groupby(vehicle_col)["totalclaims"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(12, 6))

    sns.barplot(
        x=result.index,
        y=result.values
    )

    plt.title(f"Top {top_n} Vehicle Makes by Average Claims")
    plt.xlabel(vehicle_col)
    plt.ylabel("Average Claim Amount")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

# Better plot style
sns.set_style("whitegrid")


# =========================================================
# 1. LOSS RATIO BY PROVINCE
# =========================================================

def plot_loss_ratio_by_province(df):
    """
    Creative barplot for loss ratio across provinces.
    """

    province_data = (
        df.groupby("province")[["totalclaims", "totalpremium"]]
        .sum()
    )

    province_data["LossRatio"] = (
        province_data["totalclaims"] /
        province_data["totalpremium"]
    )

    province_data = province_data.sort_values(
        by="LossRatio",
        ascending=False
    )

    plt.figure(figsize=(12, 6))

    sns.barplot(
        x=province_data.index,
        y=province_data["LossRatio"]
    )

    plt.title(
        "Loss Ratio by Province",
        fontsize=18,
        fontweight="bold"
    )

    plt.xlabel("Province", fontsize=12)
    plt.ylabel("Loss Ratio", fontsize=12)

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# =========================================================
# 2. CLAIM AMOUNT DISTRIBUTION + OUTLIERS
# =========================================================

def plot_claim_distribution(df):
    """
    Histogram + KDE for TotalClaims.
    """

    plt.figure(figsize=(12, 6))

    sns.histplot(
        df["totalclaims"],
        bins=50,
        kde=True
    )

    plt.title(
        "Distribution of Total Claims",
        fontsize=18,
        fontweight="bold"
    )

    plt.xlabel("Total Claims")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


# =========================================================
# 3. TEMPORAL CLAIM TREND ANALYSIS
# =========================================================

def plot_monthly_claim_trends(df):
    """
    Monthly claim frequency trend over time.
    """

    # Convert to datetime
    df["transactiondate"] = pd.to_datetime(
        df["transactiondate"]
    )

    # Extract year-month
    df["YearMonth"] = (
        df["transactiondate"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly_claims = (
        df.groupby("YearMonth")["totalclaims"]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(14, 6))

    sns.lineplot(
        data=monthly_claims,
        x="YearMonth",
        y="totalclaims",
        marker="o"
    )

    plt.title(
        "Monthly Total Claims Trend",
        fontsize=18,
        fontweight="bold"
    )

    plt.xlabel("Month")
    plt.ylabel("Total Claims")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# =========================================================
# 4. BONUS: VEHICLE MAKE VS CLAIMS
# =========================================================

def plot_top_vehicle_claims(df, top_n=10):
    """
    Top vehicle makes with highest average claims.
    """

    vehicle_claims = (
        df.groupby("VehicleMake")["totalclaims"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=vehicle_claims,
        x="VehicleMake",
        y="totalclaims"
    )

    plt.title(
        f"Top {top_n} Vehicle Makes by Average Claims",
        fontsize=18,
        fontweight="bold"
    )

    plt.xlabel("Vehicle Make")
    plt.ylabel("Average Claim Amount")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# =========================================================
# RUN ALL PLOTS
# =========================================================

def run_eda_visualizations(df):
    """
    Generate all EDA insight plots.
    """

    plot_loss_ratio_by_province(df)

    plot_claim_distribution(df)

    plot_monthly_claim_trends(df)

    plot_top_vehicle_claims(df)
