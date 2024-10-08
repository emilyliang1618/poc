import streamlit as st
import pandas as pd
from datetime import datetime
from io import BytesIO
import os
import plotly.express as px

# Function to determine the start date of a given quarter and year
def get_quarter_start_date(quarter, year):
    quarter_start_month = {
        'Q1': 1,
        'Q2': 4,
        'Q3': 7,
        'Q4': 10
    }[quarter]
    return datetime(year, quarter_start_month, 1)


# Function to create a roster for Lilly products
def create_lilly_roster(df, product_name, quarter, year):
    quarter_start_date = get_quarter_start_date(quarter, year)

    if product_name in ["Omvoh", "Kisunla"]:
        eff_date_col = 'Omvoh Program Effective Date'
        exp_date_col = 'Omvoh Program End Date'
    elif product_name in ["Erbitux", "Cyramza", "Retevmo", "Jaypirca"]:
        eff_date_col = f'{product_name} Program Effective Date'
        exp_date_col = f'{product_name} Program End Date'
    elif product_name == "Verzenio":
        cohort_a_parents = [
            70047, 70226, 70247, 70250, 70297, 70581, 73432, 10124985,
            100236340, 100246476, 100260940, 100376068
        ]
        cohort_b_parents = [
            70085, 70102, 70211, 70266, 70475, 70490, 73131, 73566, 186832,
            202535, 10107004, 10107291, 10133673, 100238919, 100247707,
            100261256, 100377153, 100714507, 100740511, 100742364,
            100774194, 10133633
        ]
        eff_date_col = 'Verzenio Program Effective Date'
        exp_date_col = 'Verzenio Program End Date'
    else:
        st.warning("No data available for this product under Eli Lilly.")
        return pd.DataFrame(), ""

    if eff_date_col not in df.columns or exp_date_col not in df.columns:
        st.error(f"Columns '{eff_date_col}' and/or '{exp_date_col}' are missing from the uploaded file.")
        return pd.DataFrame(), ""

    df_filtered = df.dropna(subset=[eff_date_col, exp_date_col])

    if product_name == "Verzenio":
        df_filtered['Roster Group'] = df_filtered['Group ID/Parent'].apply(
            lambda x: 'Group A' if x in cohort_a_parents else ('Group B' if x in cohort_b_parents else None)
        )
        df_filtered = df_filtered.dropna(subset=['Roster Group'])  # Remove rows with 'Unknown' group
    else:
        df_filtered['Roster Group'] = 'Unity Lilly'

    final_df = pd.DataFrame({
        'Calendar Date': quarter_start_date.strftime('%m/%d/%Y'),
        'Roster Group': df_filtered['Roster Group'],
        'Product Name': product_name,
        'Contract ID': None,
        'PARENT_ID': df_filtered['Group ID/Parent'],
        'CUSTOMER_ID': df_filtered['Parent/Child (P/C)'],
        'VALID_EFF_DATE': df_filtered[eff_date_col].dt.strftime('%m/%d/%Y'),
        'VALID_EXP_DATE': df_filtered[exp_date_col].dt.strftime('%m/%d/%Y')
    })

    output_filename = f"{product_name}_{quarter}'{str(year)[-2:]}_roster.xlsx"

    return final_df, output_filename


#Function to create a roster for Unity products and Alunbrig
def create_unity_roster(df, product_name, quarter, year):
    # Determine the start date of the specified quarter
    quarter_start_date = get_quarter_start_date(quarter, year)

    # Check if 'Start Date' and 'End Date' columns are present
    if 'Start Date' not in df.columns or 'End Date' not in df.columns:
        st.error("Columns 'Start Date' and/or 'End Date' are missing from the uploaded file.")
        return pd.DataFrame(), ""

        # Logic for Remicade and Infliximab
    if product_name.lower() in ['remicade', 'infliximab']:
        # Filter for Parent ID 70297 or 70581 and label the roster group as 'TxO and Maryland'
        df_txO_md = df[df['Parent'].isin([70297, 70581])].copy()
        df_txO_md['Roster Group'] = 'TxO and Maryland'

        # Filter for entries that are not Parent ID 70297 or 70581 and label the roster group as 'Unity Others'
        df_unity_others = df[~df['Parent'].isin([70297, 70581])].copy()
        df_unity_others['Roster Group'] = 'Unity Others'

        # Combine the two DataFrames
        new_df = pd.concat([df_txO_md, df_unity_others])

    # Logic for Jemperli
    elif product_name.lower() == 'jemperli':
        # Filter for Parent ID 70297 and label the roster group as TxO
        df_70297 = df[df['Parent'] == 70297].copy()
        df_70297['Roster Group'] = 'TxO'

        # Filter for entries that are not Parent ID 70297 and label the roster group as Unity
        df_not_70297 = df[df['Parent'] != 70297].copy()
        df_not_70297['Roster Group'] = 'Unity'

        # Combine the two DataFrames
        new_df = pd.concat([df_70297, df_not_70297])

    # Logic for Alunbrig and similar products
    elif product_name.lower() in ['alunbrig', 'bavencio', 'inlyta', 'lorbrena', 'oxbryta', 'ruxience', 'trazimera',
                                  'zirabev', 'elrexfio']:
        # Filter for Group 'ALCS' and label it as 'OM'
        df_alcs = df[df['Group'] == 'ALCS'].copy()
        df_alcs['Roster Group'] = 'OM'

        # Update product_name to 'OXBRYTA' if applicable
        if product_name.lower() == 'oxbryta':
            product_name = 'OXBRYTA'

        # Use df_alcs as the final DataFrame
        new_df = df_alcs.copy()

    # Logic for Cinvanti and Ibrance
    elif product_name.lower() in ['cinvanti', 'ibrance']:
        # Filter for Group 'ALCS' and label it as 'Performance'
        df_alcs = df[df['Group'] == 'ALCS'].copy()
        df_alcs['Roster Group'] = 'Performance'

        # Use df_alcs as the final DataFrame
        new_df = df_alcs.copy()

    # Logic for Vanflyta and Tavalisse
    elif product_name.lower() in ['vanflyta', 'tavalisse']:
        # Filter for Parent ID 70297 and label the roster group as TxO
        new_df = df[df['Parent'] == 70297].copy()
        new_df['Roster Group'] = 'TxO'

    # Logic for Nerlynx
    elif product_name.lower() == 'nerlynx':
        # Filter for Parent ID 70247 and label the roster group as RMCC
        df_rmcc = df[df['Parent'] == 70247].copy()
        df_rmcc['Roster Group'] = 'RMCC'

        # Filter for Parent ID 70297 and label the roster group as TxO
        df_txO = df[df['Parent'] == 70297].copy()
        df_txO['Roster Group'] = 'TxO'

        # Filter for entries that are not Parent ID 70247 or 70297 and label them as Performance
        df_performance = df[~df['Parent'].isin([70247, 70297])].copy()
        df_performance['Roster Group'] = 'Performance'

        # Combine all DataFrames
        new_df = pd.concat([df_rmcc, df_txO, df_performance])

    # Logic for Sustol
    elif product_name.lower() == 'sustol':
        # Filter for Parent ID 70247 and label the roster group as RMCC
        df_rmcc = df[df['Parent'] == 100714507].copy()
        df_rmcc['Roster Group'] = 'Epic Care'

        # Filter for Parent ID 70297 and label the roster group as TxO
        df_txO = df[df['Parent'] == 100740511].copy()
        df_txO['Roster Group'] = 'CCK'

         # Filter for entries that are not Parent ID 70247 or 70297 and label them as Performance
        df_performance = df[~df['Parent'].isin([100714507, 100740511])].copy()
        df_performance['Roster Group'] = 'Performance'

         # Combine all DataFrames
        new_df = pd.concat([df_rmcc, df_txO, df_performance])

    else:
        st.error("Invalid product name.")
        return pd.DataFrame(), ""

    # Prepare the final DataFrame
    final_df = pd.DataFrame({
        'Calendar Date': quarter_start_date.strftime('%m/%d/%Y'),  # Format the date without time
        'Roster Group': new_df['Roster Group'],
        'Product Name': product_name,
        'Contract ID': None,  # or pd.NA
        'PARENT_ID': new_df['Parent'],
        'CUSTOMER_ID': new_df['Ship To'],
        'VALID_EFF_DATE': new_df['Start Date'].dt.strftime('%m/%d/%Y'),
        'VALID_EXP_DATE': new_df['End Date'].apply(lambda x: x.strftime('%m/%d/%Y') if pd.notna(x) else '12/31/9999')
    })

    # Sort the DataFrame if needed
    final_df = final_df.sort_values(['Roster Group'])

    # Generate output filename
    output_filename = f"{product_name}_{quarter}'{str(year)[-2:]}_roster.xlsx"

    return final_df, output_filename

def create_injectafer_roster(unity_df, injectafer_baseline_df, quarter, year):
    st.subheader("Injectafer Roster Creation")

    injectafer_baseline_df = injectafer_baseline_df[['Parent ID', 'Program']].drop_duplicates(subset='Parent ID',
                                                                                              keep='first')

    merged_df = pd.merge(unity_df, injectafer_baseline_df, left_on='Parent', right_on='Parent ID', how='inner')
    merged_df['Roster Group'] = merged_df['Program']

    quarter_start_date = pd.to_datetime(f"{year}-{int(quarter[1]) * 3 - 2:02d}-01")

    final_df = pd.DataFrame({
        'Calendar Date': quarter_start_date.strftime('%m/%d/%Y'),
        'Roster Group': merged_df['Roster Group'],
        'Product Name': 'Injectafer',
        'Contract ID': None,
        'PARENT_ID': merged_df['Parent'],
        'CUSTOMER_ID': merged_df['Ship To'],
        'VALID_EFF_DATE': merged_df['Start Date'].dt.strftime('%m/%d/%Y'),
        'VALID_EXP_DATE': merged_df['End Date'].apply(lambda x: x.strftime('%m/%d/%Y') if pd.notna(x) else '12/31/9999')
    })

    final_df = final_df.sort_values(['Roster Group'])
    output_filename = f"Injectafer_{quarter}'{str(year)[-2:]}_roster.xlsx"

    return final_df, output_filename

# Function to convert DataFrame to Excel and return as a BytesIO object
def to_excel(df, sheet_name='Rosters'):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        date_format = workbook.add_format({'num_format': 'mm/dd/yyyy'})
        worksheet.set_column('A:A', None, date_format)
        worksheet.set_column('G:H', None, date_format)
    processed_data = output.getvalue()
    return processed_data


#Streamlit Application
st.sidebar.title("Select a POC")
page = st.sidebar.radio("", ["Eli Lilly Roster", "Genentech Roster", "Unity Roster", "NCR Analysis"])

if page == "Eli Lilly Roster":
    st.subheader("Create Roster for Eli Lilly")

    # Step 1: User Input for Quarter and Year
    quarter = st.selectbox("Select the Quarter", ["Q1", "Q2", "Q3", "Q4"])
    year = st.number_input("Enter the Year", min_value=2000, max_value=2100, value=datetime.now().year)

    # Step 2: Select Drug Product
    products = ["Omvoh", "Cyramza", "Erbitux", "Verzenio", "Retevmo", "Jaypirca", "Kisunla"]
    product_name = st.selectbox("Select the Drug Product", sorted(products))

    # Step 3: Upload Base Roster
    uploaded_file = st.file_uploader("Upload the Base Roster (Excel file)", type=["xlsx"])

    if uploaded_file is not None:
        file_name = uploaded_file.name.lower()
        if 'lilly' not in file_name:
            st.error("Please upload a valid Eli Lilly roster file. You did not upload a Lilly roster.")
            uploaded_file = None
        else:
            df = pd.read_excel(uploaded_file)
            st.success("Base roster uploaded successfully!")
            st.dataframe(df.head())

            # Step 4: Generate Roster
            if st.button("Create Roster"):
                roster_df, output_filename = create_lilly_roster(df, product_name, quarter, year)
                if not roster_df.empty:
                    st.write(f"Roster for {product_name} for {quarter} {year} has been created.")
                    num_rows, num_columns = roster_df.shape
                    st.write(f"The roster contains {num_rows} rows and {num_columns} columns.")
                    st.write("Here is a preview of the generated roster:")
                    st.dataframe(roster_df.head())
                    st.write("Click below to download the generated roster:")
                    excel_data = to_excel(roster_df)
                    st.download_button(
                        label="Download Roster",
                        data=excel_data,
                        file_name=output_filename,
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
    else:
        st.warning("Please upload a base roster file.")

elif page == "Unity Roster":
    st.subheader("Create Roster for Unity")

    # Step 1: User Input for Quarter and Year
    quarter = st.selectbox("Select the Quarter", ["Q1", "Q2", "Q3", "Q4"])
    year = st.number_input("Enter the Year", min_value=2000, max_value=2100, value=datetime.now().year)

    # Step 2: Select Drug Product
    product_name = st.selectbox(
        "Select the Drug Product",
        sorted(["Jemperli", "Alunbrig", "Vanflyta", "Tavalisse", "Bavencio", "Inlyta", "Lorbrena", "Oxbryta", "Ruxience", "Trazimera",
                "Zirabev", "Elrexfio", "Cinvanti", "Ibrance", "Sustol", "Injectafer", "Nerlynx", "Remicade", "Infliximab"])
    )

    # Step 3: Upload Base Roster
    uploaded_file = st.file_uploader("Upload the Base Roster (Excel file)", type=["xlsx"])

    if uploaded_file is not None:
        # Check if the uploaded file is appropriate for Unity products
        file_name = uploaded_file.name.lower()  # Convert file name to lower case

        if 'unity membership' not in file_name:
            st.error(
                "Please upload a valid Unity roster file. The uploaded file does not match the expected file format.")
            uploaded_file = None  # Reset the uploaded file
        else:
            df = pd.read_excel(uploaded_file)
            st.success("Base roster uploaded successfully!")
            st.dataframe(df.head())  # Display the first few rows of the dataframe

            # Check if the selected product is 'Injectafer'
            if product_name.lower() == "injectafer":
                # Step 4: Upload the additional baseline file for Injectafer
                baseline_file = st.file_uploader("Upload the Injectafer Baseline File (Excel file)", type=["xlsx"])
                if baseline_file is not None:
                    baseline_df = pd.read_excel(baseline_file)
                    st.success("Injectafer baseline file uploaded successfully!")
                    st.dataframe(baseline_df.head())  # Display the first few rows of the baseline dataframe
                else:
                    st.warning("Please upload the Injectafer baseline file.")
                    st.stop()  # Stop further processing until the baseline file is uploaded

            # Step 5: Generate Roster
            if st.button("Create Roster"):
                if product_name.lower() == "injectafer":
                    # Generate the roster using the Injectafer-specific function
                    roster_df, output_filename = create_injectafer_roster(df, baseline_df, quarter, year)
                else:
                    # Generate the roster using the standard Unity function for other Unity products
                    roster_df, output_filename = create_unity_roster(df, product_name, quarter, year)

                if not roster_df.empty:
                    st.write(f"Roster for {product_name} for {quarter} {year} has been created.")

                    # Display the number of rows and columns in the generated roster
                    num_rows, num_columns = roster_df.shape
                    st.write(f"The roster contains {num_rows} rows and {num_columns} columns.")

                    # Show example of the roster dataframe
                    st.write("Here is a preview of the generated roster:")
                    st.dataframe(roster_df.head())

                    # Step 6: Download the Roster
                    st.write("Click below to download the generated roster:")
                    excel_data = to_excel(roster_df)
                    st.download_button(
                        label="Download Roster",
                        data=excel_data,
                        file_name=output_filename,
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
    else:
        st.warning("Please upload a base roster file.")


#NEW
elif page == "NCR Analysis":

    # Load the data
    @st.cache_data
    def load_data(file_path):
        return pd.read_csv(file_path)


    # Path to the CSV file
    csv_file_path = 'data.csv'

    # Get the last modified time of the CSV file
    last_modified_time = os.path.getmtime(csv_file_path)

    # Load the data with the file path as a dependency
    df = load_data(csv_file_path)

    # Initialize session state for analysis data if it doesn't exist
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = []
    if 'rebate_tiers' not in st.session_state:
        st.session_state.rebate_tiers = []

    st.title('NCR Analysis')

    # Step 1: Select Product Type (Category)
    product_types = df['Product Type'].unique()
    selected_category = st.selectbox('Select Product Type', product_types)

    # Step 2: Select Product
    products = df[df['Product Type'] == selected_category]['Product Name'].unique()
    selected_product = st.selectbox('Select Product', products)

    # Step 3: Select GPO Type
    gpo_types = df[(df['Product Type'] == selected_category) &
                   (df['Product Name'] == selected_product)]['GPO Type'].unique()
    selected_gpo = st.selectbox('Select GPO Type', gpo_types)

    # Step 4: Select Report Year and Quarter
    years = df['Report Year'].unique()
    quarters = df['Report Quarter'].unique()
    selected_year = st.selectbox('Select Report Year', years)
    selected_quarter = st.selectbox('Select Report Quarter', quarters)

    # Step 5: Get pricing information
    if selected_product and selected_gpo:
        pricing_info = df[(df['Product Type'] == selected_category) &
                          (df['Product Name'] == selected_product) &
                          (df['GPO Type'] == selected_gpo) &
                          (df['Report Year'] == selected_year) &
                          (df['Report Quarter'] == selected_quarter)].iloc[0]

        product_usage = pricing_info['Product Usage']  # 'Oral' or 'Injectable'
        wac = pricing_info['WAC per Unit']
        contract_price = pricing_info['Contract per Unit']
        conversion_factor = pricing_info['Conversion Factor']

        if product_usage == 'Oral':
            reimbursement = pricing_info['AWP']
        else:
            reimbursement = pricing_info['ASP']

        rebate_type = pricing_info['Rebate Type']

        # Create a list of lists to display without index
        pricing_data = [
            ['Product Type', product_usage],
            ['WAC', f"${wac:,.2f}"],
            ['Contract Price', f"${contract_price:,.2f}"],
            ['Reimbursement', f"${reimbursement:,.2f}"],
            ['Rebate Type', rebate_type],
            ['Conversion Factor', f"{conversion_factor:.2f}"]
        ]

        # Display the table using st.table
        st.table(pd.DataFrame(pricing_data, columns=['Description', 'Details']).style.hide(axis="index"))

    # Step 6: Collect rebate percentages
    st.write("Enter Rebate Tiers:")
    num_rebate_tiers = st.number_input("Number of rebate tiers", min_value=1, value=1)
    rebate_tiers = []
    for i in range(num_rebate_tiers):
        tier = st.text_input(f"Rebate Tier {i + 1} (%)", value="0.0", key=f"tier_{i}")
        try:
            rebate_tiers.append(float(tier))
        except ValueError:
            st.warning(f"Invalid input for Tier {i + 1}. Please enter a numeric value.")
            rebate_tiers.append(0.0)

    # Step 7: Add to Analysis
    if st.button("Add to Analysis"):
        if selected_product and selected_gpo:
            percent_off_wac = ((wac - contract_price) / wac) * 100

            for tier in rebate_tiers:
                if rebate_type == "W":
                    rebate_amount = (tier / 100) * wac
                elif rebate_type == 'C':
                    rebate_amount = (tier / 100) * contract_price

                net_price = contract_price - rebate_amount
                net_recovery = reimbursement - net_price
                ncr_percent = (net_recovery / net_price) * 100
                nrr_percent = (net_recovery / reimbursement) * 100

                # No duplicate check; add every entry
                st.session_state.analysis_data.append({
                    "Category": selected_category,
                    "Product": selected_product,
                    "GPO": selected_gpo,
                    "Conversion Factor": f"{conversion_factor:.2f}",
                    "WAC": f"${wac:,}",
                    "Contract Price": f"${contract_price:,}",
                    "% off WAC": f"{percent_off_wac:.2f}%",
                    "Rebate Type": rebate_type,
                    "Rebate %": f"{tier:.2f}%",
                    "Rebate $": f"${rebate_amount:,.2f}",
                    "Net Price": f"${net_price:,.2f}",
                    "Reimbursement": f"${reimbursement:,.2f}",
                    "Net Recovery $": f"${net_recovery:,.2f}",
                    "NCR %": f"{ncr_percent:.2f}%",
                    "NRR %": f"{nrr_percent:.2f}%"
                })

            st.success(f"Added: {selected_product} with {selected_gpo} to analysis with a Rebate Tier of {tier:.2f}%.")
        else:
            st.warning("Please select both a product and a GPO.")

    # Step 8: Option to remove specific entries
    if st.session_state.analysis_data:
        # Create a set of unique entries for the selectbox
        unique_entries = set(
            f"{item['Product']} ({item['GPO']}) - {item.get('Rebate %', 'N/A')}"
            for item in st.session_state.analysis_data
        )

        # Convert the set back to a list for sorting
        unique_entries = list(unique_entries)


        # Define a function to extract product name and rebate percentage for sorting
        def sort_key(entry):
            # Split the entry into product, GPO, and rebate percentage
            product_gpo, rebate_str = entry.rsplit(" - ", 1)
            # Extract the product name
            product_name = product_gpo.split(" (")[0]
            # Convert rebate percentage to a float for proper numerical sorting
            rebate_percentage = float(rebate_str.replace("%", "").strip()) if rebate_str != 'N/A' else float('inf')
            return (product_name, rebate_percentage)


        # Sort the entries by product name and then by rebate percentage
        unique_entries.sort(key=sort_key)

        # Create the selectbox with the sorted entries
        entry_to_remove = st.selectbox(
            "Select an entry to remove:",
            unique_entries
        )

        if st.button("Remove Selected Entry"):
            st.session_state.analysis_data = [
                item for item in st.session_state.analysis_data
                if f"{item['Product']} ({item['GPO']}) - {item.get('Rebate %', 'N/A')}" != entry_to_remove
            ]
            st.success(f"Removed: {entry_to_remove} from analysis.")

    # Reset button to clear all analysis data
    if st.button("Start Over"):
        st.session_state.analysis_data.clear()  # Clear analysis data
        st.session_state.rebate_tiers.clear()  # Clear rebate tiers
        st.success("All analysis data has been cleared. You can start over.")

    # Step 10: Display the analysis results
    if st.session_state.analysis_data:
        st.write("Analysis Results:")

        # Convert session data to DataFrame
        analysis_df = pd.DataFrame(st.session_state.analysis_data)

        # Remove duplicate rows in the DataFrame
        analysis_df = analysis_df.drop_duplicates()

        # Convert "Rebate %" column to float for sorting
        analysis_df['Rebate %'] = analysis_df['Rebate %'].str.replace('%', '').astype(float)

        # Sort the DataFrame first by Product, then by GPO, and then by Rebate %
        analysis_df.sort_values(by=['Product', 'GPO', 'Rebate %'], ascending=[True, True, True], inplace=True)

        # Display the DataFrame without duplicates
        st.dataframe(analysis_df)

        # Add download button for the results
        csv = analysis_df.to_csv(index=False)
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name="ncr_analysis_results.csv",
            mime="text/csv",
        )
    else:
        st.write("No analysis data yet. Add products to begin the analysis.")

    st.markdown("---")

    # Step 11: What-If Analysis
    st.subheader("What-If Analysis: Adjust Reimbursement")

    # User selects the product and GPO from the analysis results
    selected_product = st.selectbox("Select Product:", analysis_df["Product"].unique())
    selected_gpo = st.selectbox("Select GPO:", analysis_df[analysis_df["Product"] == selected_product]["GPO"].unique())

    # Filter the analysis_df for the selected product and GPO to get corresponding rebate tiers and reimbursement
    filtered_analysis = analysis_df[(analysis_df["Product"] == selected_product) & (analysis_df["GPO"] == selected_gpo)]

    # Create tier labels based on the filtered data (assuming you want the unique Rebate % as tiers)
    if not filtered_analysis.empty:
        unique_tiers = filtered_analysis["Rebate %"].unique()
        tier_labels = [f"{tier}" for tier in unique_tiers]  # Use the actual rebate percentages as labels

        # Get the reimbursement value from the filtered DataFrame
        reimbursement_str = filtered_analysis["Reimbursement"].values[0]  # Get the string value
        reimbursement = float(reimbursement_str.replace('$', '').replace(',', '').strip())  # Convert to float
    else:
        tier_labels = []

    # User selects the rebate tier for adjustment based on selected product and GPO
    if tier_labels:
        selected_tier_label = st.selectbox("Select Rebate Tier to Adjust:", tier_labels)
        selected_tier_index = tier_labels.index(selected_tier_label)

        # Get original tier from filtered analysis
        original_tier = float(selected_tier_label)  # Make sure to convert to float for calculations
    else:
        st.warning("No tiers available for the selected product and GPO.")

    # Allow user to specify the range of adjustments with a slider
    min_adjustment = -10
    max_adjustment = 10
    step = 1
    selected_range = st.slider(
        "Select Reimbursement Adjustment Range (%):",
        min_value=min_adjustment,
        max_value=max_adjustment,
        value=(min_adjustment, max_adjustment),
        step=step
    )

    # Generate adjustment values based on the selected range
    selected_adjustments = list(range(selected_range[0], selected_range[1] + 1, step))


    # Function to perform the adjustments
    def perform_adjustments(selected_product, selected_gpo, original_tier, selected_adjustments, reimbursement,
                            filtered_analysis):
        ncr_results = []

        # Get net price from filtered analysis for the selected product and GPO
        net_price_str = filtered_analysis["Net Price"].values[0]  # Assuming 'Net Price' is the correct column name
        # Remove the dollar sign and convert to float
        net_price = float(net_price_str.replace('$', '').replace(',', '').strip())  # Convert to float

        for adjustment in selected_adjustments:
            # Adjust reimbursement instead of rebate percentage
            adjusted_reimbursement = reimbursement * (1 + adjustment / 100)

            # Calculate Net Recovery $ (Adjusted Reimbursement - Net Price)
            net_recovery = adjusted_reimbursement - net_price

            # Calculate NCR % (Net Recovery $ / Reimbursement * 100)
            ncr_percent = (net_recovery / reimbursement) * 100  # Use reimbursement instead of net_price

            # Append results, including Product and GPO
            ncr_results.append({
                "Product": selected_product,
                "GPO": selected_gpo,
                "Tier": selected_tier_label,
                "Original Rebate %": original_tier,
                "Adjustment (%)": adjustment,
                "Adjusted Reimbursement ($)": adjusted_reimbursement,
                "New Net Recovery ($)": net_recovery,
                "New NCR (%)": ncr_percent
            })

        return pd.DataFrame(ncr_results)


    # Button to run adjustments
    if st.button("Run Adjustment"):
        if selected_product and selected_gpo and original_tier is not None:
            ncr_df = perform_adjustments(selected_product, selected_gpo, original_tier, selected_adjustments,
                                         reimbursement, filtered_analysis)


            # Function to highlight the original reimbursement row
            def highlight_original_reimbursement(s):
                is_original = s['Adjustment (%)'] == 0
                return ['background-color: yellow' if is_original else '' for _ in s]


            # Apply the highlight function to the DataFrame
            styled_df = ncr_df.style.apply(highlight_original_reimbursement, axis=1)

            # Display the styled DataFrame
            st.dataframe(styled_df)  # Adjust height as needed
        else:
            st.warning("Please select a product, GPO, and rebate tiers to perform the What-If Analysis.")