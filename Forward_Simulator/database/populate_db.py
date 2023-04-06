import sqlite3


def populate_db(aerosols, output, gsd, diameter):
    """
    Populate sqlite3 db tables aerosol, bin_counts, and experiments

    Args:
        aerosols (List[AerosolClass]): List of AerosolClass objects to populate aerosols table with
        output (List[pandas.Dataframe]): List of simulation outputs (dataframes)
        gsd (List[float]): Gsd value for each aerosol in aerosols
        :param (List[float]) diameter: Diameter value for each experiment

    Returns:
        None
    """
    # Write aerosol in aerosols table in db
    # Write bin_counts in bin_counts table in db
    # Connect to opc.db
    # connect using postgresql instead of sqlite3
    conn = sqlite3.connect('opc.db')
    conn.execute('PRAGMA journal_mode=WAL')
    conn.isolation_level = None

    # conn = sqlite3.connect('opc.db')
    c = conn.cursor()
    # Write aerosol in aerosol table in db
    category = None
    new_aerosols = []
    for i in range(len(aerosols)):
        if category is None:
            category = aerosols[i].name
        else:
            category = category + '_' + aerosols[i].name
        # aerosol.gsd, aerosol.gm, aerosol.par_counts, aerosol.saturation_vapor_pressure, aerosol.svp_ref, aerosol.mm, aerosol.kappa, aerosol.hvap, aerosol.density
        c.execute(
            '''INSERT INTO aerosol (gsd, gm, par_counts, svp, svp_ref, mm, kappa, hvap, density,par_counts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,?)''',
            (gsd[i], diameter[i], aerosols[i].num_par, aerosols[i].saturation_vapor_pressure[0],
             aerosols[i].reference_temp, aerosols[i].molar_mass[0],
             aerosols[i].kappa[0], aerosols[i].hvap[0], aerosols[i].density[0],aerosols[i].num_par))
        new_aerosols.append(c.lastrowid)

    # Write output as an entry in bin_counts table in db with Category as category
    # For each item in output,
    new_outputs = []
    for out in output:
        # Loop through column in out
        # Out is a dataframe
        # Add category to out
        new_flows = []
        for i, row in out.iterrows():
            comm = 'INSERT INTO bin_counts ({}) VALUES ({})'.format(
                ', '.join(out.columns),
                ', '.join(['?' for j in range(len(out.columns))])
            )
            c.execute(comm, row)
            new_flows.append(c.lastrowid)
            # Add items 0-4 in new_flows as entry to experiments table
        if len(new_flows) == 5:
            c.execute(
                '''INSERT INTO experiments (flow_one, flow_two, flow_three, flow_four, flow_five,category) VALUES (?,?,?,?,?,?)''',
                (new_flows[0], new_flows[1], new_flows[2], new_flows[3], new_flows[4], category))
        elif len(new_flows) == 4:
            c.execute(
                '''INSERT INTO experiments (flow_one, flow_two, flow_three, flow_four,category) VALUES (?,?,?,?,?)''',
                (new_flows[0], new_flows[1], new_flows[2], new_flows[3], category))
        elif len(new_flows) == 3:
            c.execute(
                '''INSERT INTO experiments (flow_one, flow_two, flow_three,category) VALUES (?,?,?) ''',
                (new_flows[0], new_flows[1], new_flows[2], category))
        new_outputs.append(c.lastrowid)
    # Link new_aerosols and new_outputs
    for aerosol in new_aerosols:
        for out in new_outputs:
            c.execute('''INSERT INTO aerosol_experiments (aerosol_id, experiment_id) VALUES (?,?)''',
                      (aerosol, out))
    # Commit changes
    conn.commit()
    # Close connection
    conn.close()
