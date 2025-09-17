class DataETL:
    """Class to perform ETL (Extract, Transform, Load) operations on the Auto Businessline DataFrame."""

    import pandas as pd
    import os
    from ydata_profiling import ProfileReport

    def __init__(self, data):
        self.data = data

    def profile_analysis(self, output_folder, file_name, title="Data Profiling Report"):
        """Generate a profiling report for the current data and save it as an HTML file."""
        self.os.makedirs(output_folder, exist_ok=True)
        profile = self.ProfileReport(self.data, title=title, minimal=True)
        profile.to_file(f"{output_folder}/{file_name}")

    def transform(self, cap_dict=None):
        """Perform ETL transformations on the data. Optionally cap variables using a dictionary of {var: cap_value}."""

        # Cap variables if cap_dict is provided
        if cap_dict is not None:
            for var, cap_value in cap_dict.items():
                if var in self.data.columns:
                    self.data[var] = self.data[var].clip(upper=cap_value)
                    print(f"{var} capped at {cap_value}")

        # Group 'MCARA', 'CONVT', 'BUS', and 'RDSTR' 'veh_body' as 'Other'
        self.data.loc[
            self.data["veh_body"].isin(["MCARA", "CONVT", "BUS", "RDSTR"]), "veh_body"
        ] = "Other"

        # Assume single vehicle policy and create a vehicle count variable
        self.data["veh_cnt"] = 1

        # Add policy year
        self.data["data_segment"] = "2|inference"

        return self.data
