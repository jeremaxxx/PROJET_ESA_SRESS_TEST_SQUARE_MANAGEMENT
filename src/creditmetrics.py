import pandas as pd
import numpy as np
from itertools import product
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class CreditMetrics:
    def __init__(self, sector):
        """
        sector : objet qui contient au moins sector.annual_data et sector.quarterly_data
        """
        self.sector = sector

        # Étapes intermédiaires
        self.transition_matrix = None
        self.migration_ttc = None
        self.migration_pit = None
        self.barrier_matrix = None
        self.rho = None

        # Objet final
        self.zt = None

    # -----------------------------
    # Pipeline
    # -----------------------------
    def compute(self):
        """Lance toute la pipeline CreditMetrics"""

        # 1. Transition matrix brute
        self.transition_matrix = self._nb_transitions(self.sector.annual_data)

        # 2. Migration TTC
        self.migration_ttc = self._calculate_migration_ttc(self.transition_matrix)

        # 3. Migration PIT
        complete_monthly_counts = self._create_complete_monthly_migration_counts(self.sector.quarterly_data)
        self.migration_pit = self._create_matrice_PIT(complete_monthly_counts)
        self.migration_ttc_moyenne = self.migration_pit.groupby(level=1).mean()
        # 4. Barrières
        self.barrier_matrix = self._calculate_barrier_matrix(self.migration_ttc)

        # 5. Rho
        self.rho = self._calculate_rho(self.migration_ttc["D"])

        # 6. Construction de Zt
        self.zt = self.Zt(self.migration_pit, self.barrier_matrix, self.rho).optimize()

        return self.zt

    # -----------------------------
    # Étapes intermédiaires
    # -----------------------------
    def _nb_transitions(self, data):
        transition_counts = data.groupby(["rating", "next_rating"]).size().reset_index(name="n")
        transition_matrix = transition_counts.pivot_table(
            index="rating", columns="next_rating", values="n", fill_value=0
        )
        ordre_indices = ["AAA", "A", "BBB", "BB", "B", "C"]
        ordre_colonnes = ["AAA", "A", "BBB", "BB", "B", "C", "D"]
        return transition_matrix.reindex(index=ordre_indices, columns=ordre_colonnes).fillna(0)

    def _calculate_migration_ttc(self, transition_matrix):
        row_sums = transition_matrix.sum(axis=1)
        migration_ttc = transition_matrix.div(row_sums, axis=0)

        return migration_ttc

    def _create_complete_monthly_migration_counts(self, df):
        ratings = ["AAA", "A", "BBB", "BB", "B", "C", "D"]
        migrations_count = df.groupby(["year_quarter", "rating", "next_rating"]).size().reset_index(name="count")
        unique_months = migrations_count["year_quarter"].unique()
        all_combinations = pd.DataFrame(
            list(product(unique_months, ratings, ratings)),
            columns=["year_quarter", "rating", "next_rating"],
        )
        migrations_count_complete = pd.merge(
            all_combinations, migrations_count, on=["year_quarter", "rating", "next_rating"], how="left"
        )
        migrations_count_complete = migrations_count_complete[migrations_count_complete["rating"] != "D"].copy()
        migrations_count_complete["count"] = migrations_count_complete["count"].fillna(0).astype(int)
        migrations_count_complete["total_count"] = (
            migrations_count_complete.groupby(["year_quarter", "rating"])["count"].transform("sum").fillna(0).astype(int)
        )
        migrations_count_complete["transition_prob"] = (
            migrations_count_complete["count"] / migrations_count_complete["total_count"]
        ).fillna(0)
        return migrations_count_complete

    def _create_matrice_PIT(self, migrations_complete):
        return migrations_complete.pivot_table(
            index=["year_quarter", "rating"],
            columns="next_rating",
            values="transition_prob",
            fill_value=0,
        )

    def _calculate_barrier_matrix(self, ttc_matrix):
        barrier_matrix = np.zeros_like(ttc_matrix, dtype=float)
        for idx, (i, row) in enumerate(ttc_matrix.iterrows()):
            cumulative_sum = 0
            for j in reversed(range(len(row))):
                cumulative_sum += row.iloc[j]
                barrier_matrix[idx, j] = norm.ppf(cumulative_sum)
        barrier_matrix_df = pd.DataFrame(barrier_matrix, index=ttc_matrix.index, columns=ttc_matrix.columns)
        return barrier_matrix_df

    def _calculate_rho(self, pd_series):
        value = (1 - np.exp(-50 * pd_series)) / (1 - np.exp(-50))
        return 0.12 * value + 0.24 * (1 - value)

    # -----------------------------
    # Objet Zt
    # -----------------------------
    class Zt:
        def __init__(self, migration_pit, barrier_matrix, rho):
            self.migration_pit = migration_pit
            self.barrier_matrix = barrier_matrix
            self.rho = rho
            self.values = None
            self.probability_migrations = None # Calcule les migrations avec le Zt extrait
            self.mse_global = None
            self.mse_matrix = None

        def _calculate_pit_probabilities(self, z_t):
            ratings = self.barrier_matrix.index
            next_ratings = self.barrier_matrix.columns
            pit_matrix = pd.DataFrame(index=ratings, columns=next_ratings, dtype=float)

            for i, rating in enumerate(ratings):
                rho_i = self.rho[rating]
                for j, next_rating in enumerate(next_ratings):
                    if j == 0:
                        b_next = self.barrier_matrix.loc[rating, next_ratings[j + 1]]
                        numerator = b_next - np.sqrt(rho_i) * z_t
                        probability = 1 - norm.cdf(numerator / np.sqrt(1 - rho_i))
                    elif j == len(next_ratings) - 1:
                        b = self.barrier_matrix.loc[rating, next_rating]
                        probability = norm.cdf((b - np.sqrt(rho_i) * z_t) / np.sqrt(1 - rho_i))
                    else:
                        b = self.barrier_matrix.loc[rating, next_rating]
                        b_next = self.barrier_matrix.loc[rating, next_ratings[j + 1]]
                        term1 = norm.cdf((b - np.sqrt(rho_i) * z_t) / np.sqrt(1 - rho_i))
                        term2 = norm.cdf((b_next - np.sqrt(rho_i) * z_t) / np.sqrt(1 - rho_i))
                        probability = term1 - term2
                    pit_matrix.loc[rating, next_rating] = probability
            return pit_matrix

        def optimize(self):
            zt_by_time = {}
            all_probs = []
            for t in self.migration_pit.index.get_level_values(0).unique():
                pit_obs = self.migration_pit.loc[t].sort_index()

                def loss(z):
                    pit_theoretical = self._calculate_pit_probabilities(z[0]).sort_index()
                    aligned = pit_obs.align(pit_theoretical, join="inner", axis=1)
                    diff = aligned[0] - aligned[1]
                    return np.nansum((diff.values) ** 2)

                result = minimize(loss, x0=[0.0], bounds=[(-5, 5)], method="L-BFGS-B")
                zt_by_time[t] = result.x[0]

                recalculated_pds = self._calculate_pit_probabilities(result.x[0]).sort_index()
                probs_df = pd.concat([pit_obs.stack().rename("observed"), recalculated_pds.stack().rename("recalculated")], axis=1, join="inner").reset_index()
                probs_df["date"] = t
                all_probs.append(probs_df)

            self.values = pd.Series(zt_by_time).sort_index()
            self.probability_migrations = pd.concat(all_probs).set_index(["date", "rating", "next_rating"])
            self.mse_global, self.mse_matrix = self.mse() 
            return self

        # def _compute_mse(self):
        #     """Calcule la Mean Squared Error entre observed et recalculated."""
        #     diff = self.probability_migrations["observed"] - self.probability_migrations["recalculated"]
        #     return np.mean(diff**2)
        def mse(self):
            """Calcule la Mean Squared Error globale et par rating->next_rating."""
            df = self.probability_migrations.copy()
            df["squared_error"] = (df["observed"] - df["recalculated"])**2

            # MSE global
            mse_global = df["squared_error"].mean()

            # MSE par rating -> next_rating
            mse_matrix = df.groupby(["rating", "next_rating"])["squared_error"].mean().unstack().fillna(0)

            return mse_global, mse_matrix

        
        def plotting_zt(self):
            plt.figure(figsize=(10, 4))
            self.values.plot(title="Facteur systémique Zt (CreditMetrics)")
            plt.show()

        def plot_transitions(self, pairs):
            """
            Trace les courbes observed vs recalculated pour une ou plusieurs paires (rating, next_rating).

            Args:
                pairs (list of [rating, next_rating]): liste de couples.
                                                    Exemple: [["BBB","BB"], ["A","BBB"]]
            """
            if self.probability_migrations is None:
                raise ValueError("Les probabilités n'ont pas encore été calculées. Lancez optimize() d'abord.")

            import matplotlib.pyplot as plt
            from itertools import cycle

            plt.figure(figsize=(10, 5))
            colors = cycle(plt.cm.tab10.colors)  # palette cyclique pour les couleurs

            for rating, next_rating in pairs:
                color = next(colors)
                subset = self.probability_migrations.xs(
                    (rating, next_rating), level=("rating", "next_rating")
                )
                plt.plot(subset.index, subset["observed"], label=f"Observed {rating}->{next_rating}", color=color)
                plt.plot(subset.index, subset["recalculated"], linestyle="--", label=f"Recalculated {rating}->{next_rating}", color=color)

            plt.title("Probabilités Observed vs Recalculated")
            plt.xlabel("Date")
            plt.ylabel("Transition Probability")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()


        def __repr__(self):
            return f"Zt values:\n{self.values}"
