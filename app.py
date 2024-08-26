import streamlit as st
import pandas as pd
from datetime import datetime
from io import BytesIO
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
        df_unity_others['Roster Group'] = 'Unity'

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
    elif product_name.lower() in ['alunbrig', 'bavencio', 'inlyta', 'lorbrena', 'oxbryta', 'ruxience', 'trazimera', 'zirabev', 'elrexfio']:
        # Filter for Group 'ALCS' and label it as 'OM'
        df_alcs = df[df['Group'] == 'ALCS'].copy()
        df_alcs['Roster Group'] = 'OM'

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
page = st.sidebar.radio("", ["Eli Lilly Roster", "Genentech Roster", "Unity Roster", "NCR Analysis", "ASP Prediction"])

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

    st.title("NCR Analysis Test")

    # Define the GPOs
    gpos = [
        "Onmark Standard",
        "Onmark United",
        "Perform",
        "Perform Plus",
        "Unity",
        "Unity USON"
    ]

    # Sample data
    pricing_data = {
        "Ibrance IBRANCE 125MG CAPSULE 21/EA": {
            "WAC": 15982,
            "AWP": 19178.87,
            "Contract Prices": {gpo: 14544 for gpo in gpos}
        },
        "Kisqali KISQALI 600MG DAILY TAB 3X21 BLST 63/EA": {
            "WAC": 17685,
            "AWP": 21222.44,
            "Contract Prices": {gpo: 15917 for gpo in gpos}
        },
        "Verzenio VERZENIO 150MG TAB BLSTR 14/EA": {
            "WAC": 3851,
            "AWP": 4621.72,
            "Contract Prices": {
                "Onmark Standard": 3505,
                **{gpo: 3466 for gpo in gpos if gpo != "Onmark Standard"}
            }
        },
        "Verzenio VERZENIO 150MG TAB BLSTR 14/EA X 4": {
            "WAC": 3851 * 4,
            "AWP": 4621.72 * 4,
            "Contract Prices": {
                "Onmark Standard": 3505 * 4,
                **{gpo: 3466 * 4 for gpo in gpos if gpo != "Onmark Standard"}
            }
        }
    }

    # Define the categories
    categories = [
        "CDK 4/6 Orals",
        "Biosim",
        "BTK",
        "IO",
        "Ultomiris/Soliris",
        "Monoferric"
    ]

    # Initialize session state for analysis data if not already initialized
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = []

    # User selects a category
    selected_category = st.selectbox("Select a category:", categories)

    # Products for the selected category
    if selected_category == "CDK 4/6 Orals":
        products = list(pricing_data.keys())
    else:
        products = ["Product 1", "Product 2"]  # Placeholder for other categories

    # User selects a product
    selected_product = st.selectbox("Select a product:", products)

    # User selects a single GPO
    selected_gpo = st.selectbox("Select a GPO for the product:", gpos)

    # Allow the user to specify the rebate type (WAC or Contract Price)
    rebate_type = st.selectbox("Rebate Type (W or C):", ["W", "C"])

    # Allow the user to specify the number of rebate tiers
    num_rebate_tiers = st.number_input("Number of Rebate Tiers:", min_value=1, max_value=5, step=1)

    # Collect rebate percentages using text input
    rebate_tiers = []
    for i in range(num_rebate_tiers):
        tier = st.text_input(f"Rebate Tier {i + 1} (%)", value="10.0")
        try:
            rebate_tiers.append(float(tier))
        except ValueError:
            st.warning(f"Invalid input for Tier {i + 1}. Please enter a numeric value.")
            rebate_tiers.append(0.0)

    # Get the relevant pricing data
    if selected_product and selected_gpo:
        wac = pricing_data[selected_product]["WAC"]
        contract_price = pricing_data[selected_product]["Contract Prices"][selected_gpo]
        awp = pricing_data[selected_product]["AWP"]

        # Calculate Reimbursement (AWP - 19.5%)
        reimbursement = awp - (0.195 * awp)

    # Add the selection to the analysis data
    if st.button("Add to Analysis"):
        if selected_product and selected_gpo:
            # Calculate the % off WAC to get to the Contract Price
            percent_off_wac = ((wac - contract_price) / wac) * 100

            # Add each rebate tier as a separate row
            for tier in rebate_tiers:
                if rebate_type == "W":
                    rebate_amount = (tier / 100) * wac
                elif rebate_type == 'C':
                    rebate_amount = (tier / 100) * contract_price

                # Calculate Net Price (Contract Price - Rebate $)
                net_price = contract_price - rebate_amount

                # Calculate Net Recovery $ (Net Price - Reimbursement)
                net_recovery = reimbursement - net_price

                # Calculate NCR % (Net Recovery $ / Net Price * 100)
                ncr_percent = (net_recovery / net_price) * 100

                # Calculate NRR % (Net Recovery $ / Reimbursement * 100)
                nrr_percent = (net_recovery / reimbursement) * 100

                st.session_state.analysis_data.append({
                    "Category": selected_category,
                    "Product": selected_product,
                    "GPO": selected_gpo,
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

            st.success(f"Added: {selected_product} with {selected_gpo} to analysis!")
        else:
            st.warning("Please select both a product and a GPO.")

    # Option to remove specific entries
    if st.session_state.analysis_data:
        entry_to_remove = st.selectbox(
            "Select an entry to remove:",
            [f"{item['Product']} ({item['GPO']}) - {item.get('Rebate %', 'N/A')}" for item in
             st.session_state.analysis_data]
        )

        if st.button("Remove Selected Entry"):
            st.session_state.analysis_data = [
                item for item in st.session_state.analysis_data
                if f"{item['Product']} ({item['GPO']}) - {item.get('Rebate %', 'N/A')}" != entry_to_remove
            ]
            st.success(f"Removed: {entry_to_remove} from analysis.")

    # Reset the analysis data (placed before the DataFrame)
    if st.button("Reset Analysis"):
        st.session_state.analysis_data = []
        st.success("Analysis data reset!")

    # Convert the analysis data to a DataFrame
    analysis_df = pd.DataFrame(st.session_state.analysis_data)

    # Display the DataFrame
    st.subheader("Analysis Preview")
    st.dataframe(analysis_df)

    # Dynamic Reimbursement Adjustment Section
    st.subheader("What-If Analysis: Adjust Reimbursement")

    # User selects the rebate tier for adjustment
    tier_labels = [f"Tier {i + 1}" for i in range(num_rebate_tiers)]
    selected_tier_label = st.selectbox("Select Rebate Tier to Adjust:", tier_labels)
    selected_tier_index = tier_labels.index(selected_tier_label)

    # Allow user to specify the range of adjustments with a slider
    min_adjustment = -15
    max_adjustment = 15
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

    # Calculate the new NCR% for each adjustment
    if selected_product and selected_gpo and rebate_tiers:
        ncr_results = []
        original_tier = rebate_tiers[selected_tier_index]
        selected_tier_label = f"Tier {selected_tier_index + 1}"  # Get the tier label

        for adjustment in selected_adjustments:
            # Adjust reimbursement instead of rebate percentage
            adjusted_reimbursement = reimbursement * (1 + adjustment / 100)

            # Calculate Net Price (Contract Price - Rebate $)
            if rebate_type == "W":
                rebate_amount = (original_tier / 100) * wac
            elif rebate_type == 'C':
                rebate_amount = (original_tier / 100) * contract_price
            net_price = contract_price - rebate_amount

            # Calculate Net Recovery $ (Net Price - Adjusted Reimbursement)
            net_recovery = adjusted_reimbursement - net_price

            # Calculate NCR % (Net Recovery $ / Net Price * 100)
            ncr_percent = (net_recovery / net_price) * 100

            ncr_results.append({
                "Tier": selected_tier_label,
                "Original Rebate %": original_tier,
                "Adjustment (%)": adjustment,
                "Adjusted Reimbursement ($)": adjusted_reimbursement,
                "New NCR (%)": ncr_percent
            })

        # # Convert the results to a DataFrame
        # ncr_df = pd.DataFrame(ncr_results)
        # ncr_df = ncr_df.sort_values(by="New NCR (%)")
        #
        #
        # # Display the DataFrame
        # st.subheader("NCR Results for Selected Rebate Tier")
        # st.dataframe(ncr_df)

        # Convert the results to a DataFrame
        ncr_df = pd.DataFrame(ncr_results)
        ncr_df = ncr_df.sort_values(by="New NCR (%)")

        # Function to highlight the original reimbursement row
        def highlight_original_reimbursement(s):
            is_original = s['Adjustment (%)'] == 0
            return ['background-color: yellow' if is_original else '' for _ in s]

       # Apply the highlight function to the DataFrame
        styled_df = ncr_df.style.apply(highlight_original_reimbursement, axis=1)
        st.dataframe(styled_df, height=800)  # Adjust height as needed

    # Generate plot data for all tiers
    plot_data = []
    for i, rebate in enumerate(rebate_tiers):
        original_tier = rebate
        tier_label = f"Tier {i + 1}"  # Get the tier label

        for adjustment in selected_adjustments:
            adjusted_reimbursement = reimbursement * (1 + adjustment / 100)

            # Calculate Net Price (Contract Price - Rebate $)
            if rebate_type == "W":
                rebate_amount = (original_tier / 100) * wac
            elif rebate_type == 'C':
                rebate_amount = (original_tier / 100) * contract_price
            net_price = contract_price - rebate_amount

            # Calculate Net Recovery $ (Net Price - Adjusted Reimbursement)
            net_recovery = adjusted_reimbursement - net_price

            # Calculate NCR % (Net Recovery $ / Net Price * 100)
            ncr_percent = (net_recovery / net_price) * 100

            plot_data.append({
                "Tier": tier_label,
                "Adjusted Reimbursement ($)": adjusted_reimbursement,
                "NCR %": ncr_percent
            })

    # Convert the plot data to a DataFrame
    plot_df = pd.DataFrame(plot_data)

    # Plot the data
    fig = px.line(
        plot_df,
        x="Adjusted Reimbursement ($)",
        y="NCR %",
        color="Tier",  # Use the 'Tier' column for coloring
        markers=True,
        title="NCR % vs. Adjusted Reimbursement by Rebate Tier",
        labels={"Adjusted Reimbursement ($)": "Adjusted Reimbursement ($)", "New NCR (%)": "New NCR (%)"}
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

if page == "ASP Prediction":
    st.subheader("ASP Prediction Test with FB Prophet")