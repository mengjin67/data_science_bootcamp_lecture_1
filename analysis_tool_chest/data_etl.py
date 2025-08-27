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
        # Convert 'date_of_birth' to datetime, coercing errors to NaT for invalid formats
        self.data['date_of_birth'] = self.pd.to_datetime(self.data['date_of_birth'], format='%m/%d/%Y', errors='coerce')

        # Define bins and labels for age categories
        bins = [
            self.pd.Timestamp('1900-01-01'), 
            self.pd.Timestamp('1950-01-01'), 
            self.pd.Timestamp('1960-01-01'),
            self.pd.Timestamp('1970-01-01'), 
            self.pd.Timestamp('1980-01-01'), 
            self.pd.Timestamp('1990-01-01'), 
            self.pd.Timestamp.max 
        ]
        labels = [1, 2, 3, 4, 5, 6]
        # Use pd.cut to assign agecat2 based on date_of_birth
        self.data['agecat2'] = self.pd.cut(self.data['date_of_birth'], bins=bins, labels=labels, right=False)
        # Ensure float type for consistency
        self.data['agecat2'] = self.data['agecat2'].astype(float)
        # Group 'MCARA', 'CONVT', 'BUS', and 'RDSTR' 'veh_body' as 'Other'
        self.data.loc[self.data['veh_body'].isin(['MCARA','CONVT','BUS','RDSTR']), 'veh_body'] = 'Other'

        # Cap variables if cap_dict is provided
        if cap_dict is not None:
            for var, cap_value in cap_dict.items():
                if var in self.data.columns:
                    self.data[var] = self.data[var].clip(upper=cap_value)
                    print(f"{var} capped at {cap_value}")

        # Assume single vehicle policy and create a vehicle count variable
        self.data['veh_cnt'] = 1

        # Add policy year 
        self.data['pol_year'] = 2018

        return self.data