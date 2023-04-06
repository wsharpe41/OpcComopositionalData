import psycopg2
import configparser

config = configparser.ConfigParser()
config.read('database.ini')

host = config.get('postgresql', 'host')
database = config.get('postgresql', 'database')
user = config.get('postgresql', 'user')
password = config.get('postgresql', 'password')
port = config.get('postgresql', 'port')

def populate_db(aerosols, output, gsd, diameter):
    """
    Populate PostgreSQL db tables aerosol, bin_counts, and experiments

    Args:
        aerosols (List[AerosolClass]): List of AerosolClass objects
        output (List[pandas.Dataframe]): List of simulation outputs (dataframes)
        gsd (List[float]): Gsd value for each aerosol in aerosols
        diameter (List[float]): Diameter value for each experiment

    Returns:
        None
    """
    # Write aerosol in aerosols table in db
    # Write bin_counts in bin_counts table in db
    # Connect to opc.db
    # connect using postgresql instead of sqlite3
    conn = psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password,
        port=port
    )
    conn.set_session(autocommit=True)

    # conn = sqlite3.connect('opc.db')
    # Write aerosol in aerosol table in db
    category = None
    new_aerosols = []
    aerosol_cursor = conn.cursor()
    for i in range(len(aerosols)):
        num_par = aerosols[i].num_par
        if num_par is None:
            num_par = 1000
        if category is None:
            category = aerosols[i].name
        else:
            category = category + '_' + aerosols[i].name
        # aerosol.gsd, aerosol.gm, aerosol.par_counts, aerosol.saturation_vapor_pressure, aerosol.svp_ref, aerosol.mm, aerosol.kappa, aerosol.hvap, aerosol.density
        aerosol_cursor.execute(
            '''INSERT INTO aerosol (gsd, gm, par_counts, svp, svp_ref, mm, kappa, hvap, density) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING aerosol_id''',
            (gsd[i], diameter[i], num_par, aerosols[i].saturation_vapor_pressure[0],
             aerosols[i].reference_temp, aerosols[i].molar_mass[0],
             aerosols[i].kappa[0], aerosols[i].hvap[0], aerosols[i].density[0]))
        new_aerosols.append(aerosol_cursor.fetchone()[0])
        aerosol_cursor.close()
        aerosol_cursor = conn.cursor()
    # Write output as an entry in bin_counts table in db with Category as category
    # For each item in output,
    new_outputs = []
    for out in output:
        # Loop through column in out
        # Out is a dataframe
        # Add category to out
        new_flows = []
        for i, row in out.iterrows():
            comm = 'INSERT INTO bin_counts ({}) VALUES ({}) RETURNING bin_id'.format(
                ', '.join(out.columns),
                ', '.join(['%s' for j in range(len(out.columns))])
            )
            flow_cursor = conn.cursor()
            flow_cursor.execute(comm, row)
            new_flows.append(flow_cursor.fetchone()[0])
            flow_cursor.close()
            # Add items 0-4 in new_flows as entry to experiments table
        if len(new_flows) == 5:
            flow_cursor = conn.cursor()
            flow_cursor.execute(
                '''INSERT INTO experiments (flow_one, flow_two, flow_three, flow_four, flow_five,category) VALUES (%s,%s,%s,%s,%s,%s) RETURNING experiment_id''',
                (new_flows[0], new_flows[1], new_flows[2], new_flows[3], new_flows[4], category))
            new_outputs.append(flow_cursor.fetchone()[0])
            flow_cursor.close()
        elif len(new_flows) == 4:
            flow_cursor = conn.cursor()
            flow_cursor.execute(
                '''INSERT INTO experiments (flow_one, flow_two, flow_three, flow_four,category) VALUES (%s,%s,%s,%s,%s) RETURNING experiment_id''',
                (new_flows[0], new_flows[1], new_flows[2], new_flows[3], category))
            new_outputs.append(flow_cursor.fetchone()[0])
            flow_cursor.close()
        elif len(new_flows) == 3:
            flow_cursor = conn.cursor()
            flow_cursor.execute(
                '''INSERT INTO experiments (flow_one, flow_two, flow_three,category) VALUES (%s,%s,%s) RETURNING experiment_id''',
                (new_flows[0], new_flows[1], new_flows[2], category))
            new_outputs.append(flow_cursor.fetchone()[0])
            flow_cursor.close()
    # Link new_aerosols and new_outputs
    for aerosol in new_aerosols:
        for out in new_outputs:
            aero_exp_cursor = conn.cursor()
            aero_exp_cursor.execute('''INSERT INTO aerosol_experiments (aerosol_id, experiment_id) VALUES (%s,%s)''',
                                    (aerosol, out))
            aero_exp_cursor.close()
    # Commit changes
    conn.commit()
    # Close connection
    conn.close()
