import pandas as pd
import numpy as np
import math
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


class FantasyProjections:
    """
    Main class for generating fantasy baseball auction values.
    
    Features:
    - Dynamic column detection (no hardcoded stat lists)
    - Auto-discovers available projection systems
    - Smart qualifier detection based on percentiles
    - Flexible configuration with sensible defaults
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        data_dir: str = "data",
        year: Optional[int] = None,
        n_teams: int = 12,
        roster_size: int = 23,
        budget_per_team: int = 260,
        hitter_budget_split: float = 0.7
    ):
        """
        Initialize fantasy projections system.
        
        Args:
            data_dir: Directory containing projection and stats files
            year: Projection year (defaults to current year)
            n_teams: Number of teams in league
            roster_size: Players per team
            budget_per_team: Auction budget per team
            hitter_budget_split: Fraction of budget for hitters (vs pitchers)
        """
        # Configuration
        self.data_dir = Path(data_dir)
        self.year = year or datetime.now().year
        self.n_teams = n_teams
        self.roster_size = roster_size
        self.budget_per_team = budget_per_team
        self.hitter_budget_split = hitter_budget_split
        
        # League totals
        self.total_budget = n_teams * budget_per_team
        self.total_players = n_teams * roster_size
        
        # Position requirements (as multipliers of n_teams)
        self.position_requirements = {
            'C': 1.0,
            '1B': 1.5,
            '2B': 1.5,
            '3B': 1.5,
            'SS': 1.5,
            'OF': 5.0,
            'MI': 1.0,
            'CI': 1.0,
            'DH': 2.0,
            'SP': 6.5,
            'RP': 2.5,
            'P': 9.0,
        }
        
        # Projection systems to use
        self.proj_systems = ['cbs', 'atc', 'thebatx', 'oopsy', 'fangraphsdc', 'steamer', 'zips']

        # Position hierarchy (for determining primary position)
        self.position_hierarchy = ['C', '3B', '2B', 'OF', 'SS', '1B', 'DH', 'SP', 'RP', 'P']
        
        # Data storage
        self.player_index = None
        self.hitting_projections = None
        self.pitching_projections = None
        self.qualifiers = None
        self.final_values = None
        
        # Column name variations we know about
        self.column_mappings = {
            'IP': ['INNs', 'IP'],
            'SV': ['S', 'SV', 'Saves'],
            'HLD': ['HD', 'HLD', 'Holds'],
            'BA': ['AVG', 'BA', 'AVE'],
            'SO': ['K', 'SO'],
        }

        # Database connection
        if db_path is None:
            db_path = self.data_dir.parent.parent / "fantasy_data.db"
        self.db_path = Path(db_path)
        self.engine = create_engine(f'sqlite:///{self.db_path}') if self.db_path else None
        self.player_index = self.load_player_index()
    
    # ========================================================================
    # Main Workflow Methods
    # ========================================================================
    
    # ========================================================================
    # MODE 1: PRE-SEASON DRAFT PREPARATION
    # ========================================================================
    
    def generate_auction_values(
        self,
        previous_hitting: Optional[pd.DataFrame] = None,
        previous_pitching: Optional[pd.DataFrame] = None,
        force_reload: bool = False
    ) -> pd.DataFrame:
        """
        MODE 1: Pre-season draft preparation.
        
        Generates auction values for upcoming season using:
        - Current year projections as input
        - Previous year stats as baseline/qualifiers
        - Positional adjustments applied
        
        Args:
            previous_hitting: Previous season hitting stats (for qualifiers)
            previous_pitching: Previous season pitching stats (for qualifiers)
            force_reload: Force reload of projection data
            
        Returns:
            DataFrame with auction values for draft
        """
        logger.info("=" * 60)
        logger.info("MODE 1: GENERATING AUCTION VALUES (DRAFT PREP)")
        logger.info("=" * 60)
        print("=" * 60)
        print("MODE 1: GENERATING AUCTION VALUES (DRAFT PREP)")
        print("=" * 60)

        # Load projection data
        hitting_proj = self.load_hitting_projections()
        pitching_proj = self.load_pitching_projections()
        
        # Load previous season data if not provided
        if previous_hitting is None:
            previous_hitting = self.load_previous_season_stats('hitting')
        if previous_pitching is None:
            previous_pitching = self.load_previous_season_stats('pitching')
        
        # Calculate qualifiers from PREVIOUS season
        qualifiers = self.calculate_qualifiers(previous_hitting, previous_pitching)
        self.draft_qualifiers = qualifiers  # Save for later comparison
        self.qualifiers = qualifiers  # Set for calculate_z_scores()

        logger.info(f"Calculated qualifiers from {self.year - 1} season")
        print(f"Calculated qualifiers from {self.year - 1} season")
        
        # Calculate z-scores
        hitting_z = self.calculate_z_scores(hitting_proj, 'hitting')
        pitching_z = self.calculate_z_scores(pitching_proj, 'pitching')
        
        # Apply positional adjustments
        hitting_adj = self.apply_positional_adjustments(hitting_z, 'hitting')
        pitching_adj = self.apply_positional_adjustments(pitching_z, 'pitching')
        
        # Convert to auction values
        draft_values = self._convert_to_dollars(
            hitting_adj, 
            pitching_adj,
            z_column='z'  # Use adjusted z-scores
        )
        
        # Save conversion factor for later reference
        self.draft_conversion_factor = self._last_conversion_factor
        
        logger.info(f"Draft conversion factor: ${self.draft_conversion_factor:.2f}/z")
        logger.info("=" * 60)
        print(f"Draft conversion factor: ${self.draft_conversion_factor:.2f}/z")
        print("=" * 60)
        
        return draft_values
    
    # ========================================================================
    # MODE 2: POST-SEASON RETROSPECTIVE
    # ========================================================================
    
    def evaluate_season_performance(
        self,
        actual_hitting: pd.DataFrame,
        actual_pitching: pd.DataFrame,
        normalize_to_budget: bool = True
    ) -> pd.DataFrame:
        """
        MODE 2: Post-season full year retrospective.
        
        Evaluates actual season performance using:
        - Actual season stats as input
        - SAME previous year baseline as draft (for comparison)
        - NO positional adjustments (measuring absolute performance)
        - Normalized to league budget for comparability
        
        Args:
            actual_hitting: Actual season hitting stats
            actual_pitching: Actual season pitching stats
            normalize_to_budget: Scale to league budget (recommended: True)
            
        Returns:
            DataFrame with actual season values (comparable to draft values)
        """
        logger.info("=" * 60)
        logger.info("MODE 2: EVALUATING SEASON PERFORMANCE")
        logger.info("=" * 60)
        print("=" * 60)
        print("MODE 2: EVALUATING SEASON PERFORMANCE")
        print("=" * 60)

        if self.draft_qualifiers is None:
            raise ValueError(
                "Must run generate_auction_values() first to establish baseline. "
                "Or manually set self.draft_qualifiers."
            )
        
        # Use SAME qualifiers as draft (previous season)
        self.qualifiers = self.draft_qualifiers
        logger.info(f"Using {self.year - 1} season as baseline (same as draft)")
        print(f"Using {self.year - 1} season as baseline (same as draft)")
        
        # Calculate z-scores (NO positional adjustment)
        hitting_z = self.calculate_z_scores(actual_hitting, 'hitting')
        pitching_z = self.calculate_z_scores(actual_pitching, 'pitching')
        
        # Convert to dollars
        season_values = self._convert_to_dollars(
            hitting_z,
            pitching_z,
            z_column='total_z',  # Use unadjusted z-scores
            normalize_to_budget=normalize_to_budget
        )
        
        logger.info(f"Season conversion factor: ${self._last_conversion_factor:.2f}/z")
        logger.info(f"Draft conversion factor was: ${self.draft_conversion_factor:.2f}/z")
        logger.info("=" * 60)
        
        return season_values
    
    # ========================================================================
    # MODE 3: PERIOD PERFORMANCE (WEEKLY, MONTHLY, ETC)
    # ========================================================================
    
    def evaluate_period_performance(
        self,
        period_hitting: pd.DataFrame,
        period_pitching: pd.DataFrame,
        period_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        MODE 3: Period-based performance evaluation.
        
        Evaluates any time period (week, month, date range) using:
        - That period's stats as BOTH input AND baseline
        - All MLB players as comparison pool
        - NO positional adjustments
        - Dollar values represent "if sustained for full season"
        
        Args:
            period_hitting: Period hitting stats (Week 12, June, etc.)
            period_pitching: Period pitching stats
            period_name: Descriptive name (e.g., "Week 12", "June 2025")
            
        Returns:
            DataFrame with period values
        """
        logger.info("=" * 60)
        logger.info(f"MODE 3: EVALUATING PERIOD PERFORMANCE - {period_name or 'Custom Period'}")
        logger.info("=" * 60)
        print("=" * 60)
        print(f"MODE 3: EVALUATING PERIOD PERFORMANCE - {period_name or 'Custom Period'}")
        print("=" * 60)
        
        # Calculate qualifiers from THIS period's data
        # Use very lenient thresholds - include anyone who played
        period_qualifiers = self.calculate_qualifiers(
            period_hitting,
            period_pitching,
            min_pa=1,  # Anyone with a plate appearance
            min_ip=0.1,  # Anyone who pitched
            hitting_percentile=0.0,  # Don't filter by percentile
            pitching_percentile=0.0,
            sp_min_ip=1,  # Very low threshold for period data
            rp_ip_range=[0.1, 999]  # Essentially no filter
        )
        
        logger.info(f"Using {period_name or 'this period'} as its own baseline")
        
        # Set as current qualifiers
        period_qualifiers = self.calculate_qualifiers(...)
        self.qualifiers = period_qualifiers  # Set for calculate_z_scores()
        
        # Calculate z-scores (NO positional adjustment)
        hitting_z = self.calculate_z_scores(period_hitting, 'hitting')
        pitching_z = self.calculate_z_scores(period_pitching, 'pitching')
        
        # Convert to dollars
        period_values = self._convert_to_dollars(
            hitting_z,
            pitching_z,
            z_column='total_z',  # Use unadjusted z-scores
            normalize_to_budget=True
        )
        
        logger.info(f"Period conversion factor: ${self._last_conversion_factor:.2f}/z")
        logger.info("=" * 60)
        print(f"Period conversion factor: ${self._last_conversion_factor:.2f}/z")
        print("=" * 60)
        
        return period_values
    
    # ========================================================================
    # Data Loading Methods
    # ========================================================================
    
    def discover_projection_systems(self) -> list[str]:
        """
        Auto-discover available projection systems from files.
        
        Returns:
            List of projection system names
        """
        hitting_files = self.data_dir.glob(f"{self.year}-*-proj-h.csv")
        
        systems = []
        for file in hitting_files:
            # Extract system name from filename
            # e.g., "2025-steamer-proj-h.csv" -> "steamer"
            parts = file.stem.split('-')
            if len(parts) >= 3:
                system_name = '-'.join(parts[1:-2])  # Handle multi-word systems
                systems.append(system_name)
        
        if not systems:
            logger.warning(f"No projection files found in {self.data_dir}")
        else:
            logger.info(f"Discovered {len(systems)} projection systems: {systems}")
        
        return systems
    
    def load_hitting_projections(self, discover_systems=False) -> pd.DataFrame:
        """Load and average hitting projections from all available systems."""
        systems = self.discover_projection_systems() if discover_systems else self.proj_systems
        
        if not systems:
            raise FileNotFoundError(
                f"No projection files found in {self.data_dir}. "
                f"Looking for files like: {self.year}-[system]-proj-h.csv"
            )
        
        all_projections = []
        
        for system in systems:
            filepath = self.data_dir / f"{self.year}-{system}-proj-h.csv"
            try:
                df = pd.read_csv(filepath, encoding='latin-1')
                df['projection_system'] = system
                df = self.standardize_columns(df, 'hitting')
                all_projections.append(df)
                logger.info(f"Loaded {len(df)} hitting projections from {system}")
                print(f"Loaded {len(df)} hitting projections from {system}")
            except Exception as e:
                logger.warning(f"Could not load {system}: {e}")
        
        if not all_projections:
            raise ValueError("No projection systems loaded successfully")
        
        # Combine all projections
        all_projections = [system for system in all_projections if len(system)>0]
        combined = pd.concat(all_projections, ignore_index=True)
        
        # MAP TO CBS IDs BEFORE AVERAGING (important!)
        missing_cbsid = combined.loc[combined['cbsid'].isna()].reset_index(drop=True)
        has_cbsid = combined.loc[combined['cbsid'].notna()].reset_index(drop=True)
        merged_cbsid = missing_cbsid.merge(self.player_index[['cbsid', 'CBSNAME', 'IDFANGRAPHS', 'IDFANGRAPHS_minors']], left_on='playerid', right_on='IDFANGRAPHS', how='inner', suffixes=[None,'_y'])
        merged_cbsid = merged_cbsid.fillna({'cbsid':merged_cbsid['cbsid_y'], 'CBSNAME':merged_cbsid['CBSNAME_y']})
        merged_cbsid.drop(columns=['CBSNAME_y', 'cbsid_y', 'IDFANGRAPHS', 'IDFANGRAPHS_minors'],inplace=True)

        # Combine back with existing cbsid records
        combined = pd.concat([has_cbsid, merged_cbsid], ignore_index=True)

        # Now average by cbsid
        averaged = self.average_projections(combined, 'hitting')
        
        # Process further
        processed = self.process_hitting_projections(averaged)
        
        return processed
    
    def load_pitching_projections(self, discover_systems=False) -> pd.DataFrame:
        """Load and average pitching projections from all available systems."""
        systems = self.discover_projection_systems() if discover_systems else self.proj_systems
        
        if not systems:
            raise FileNotFoundError(
                f"No projection files found in {self.data_dir}. "
                f"Looking for files like: {self.year}-[system]-proj-p.csv"
            )
        
        all_projections = []
        
        for system in systems:
            filepath = self.data_dir / f"{self.year}-{system}-proj-p.csv"
            try:
                df = pd.read_csv(filepath, encoding='latin-1')
                df['projection_system'] = system
                df = self.standardize_columns(df, 'pitching')
                all_projections.append(df)
                logger.info(f"Loaded {len(df)} pitching projections from {system}")
            except Exception as e:
                logger.warning(f"Could not load {system}: {e}")
        
        if not all_projections:
            raise ValueError("No projection systems loaded successfully")
        
        # Combine all projections
        all_projections = [system for system in all_projections if len(system)>0]
        combined = pd.concat(all_projections, ignore_index=True)

        # MAP TO CBS IDs BEFORE AVERAGING (important!)
        missing_cbsid = combined.loc[combined['cbsid'].isna()].reset_index(drop=True)
        has_cbsid = combined.loc[combined['cbsid'].notna()].reset_index(drop=True)
        merged_cbsid = missing_cbsid.merge(self.player_index[['cbsid', 'CBSNAME', 'IDFANGRAPHS', 'IDFANGRAPHS_minors']], left_on='playerid', right_on='IDFANGRAPHS', how='inner', suffixes=[None,'_y'])
        merged_cbsid = merged_cbsid.fillna({'cbsid':merged_cbsid['cbsid_y'], 'CBSNAME':merged_cbsid['CBSNAME_y']})
        merged_cbsid.drop(columns=['CBSNAME_y', 'cbsid_y', 'IDFANGRAPHS', 'IDFANGRAPHS_minors'],inplace=True)

        # Combine back with existing cbsid records
        combined = pd.concat([has_cbsid, merged_cbsid], ignore_index=True)
        
        # Now average by cbsid
        averaged = self.average_projections(combined, 'pitching')
        
        # Process further
        processed = self.process_pitching_projections(averaged)
        
        return processed
    
    def load_previous_season_stats(
        self, 
        player_type: Literal['hitting', 'pitching']
    ) -> pd.DataFrame:
        """
        Load previous season stats for qualifier calculation.
        
        Args:
            player_type: 'hitting' or 'pitching'
            
        Returns:
            DataFrame with previous season stats
        """
        suffix = 'h' if player_type == 'hitting' else 'p'
        filepath = self.data_dir / f"{self.year - 1}-final-stats-{suffix}.csv"
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"Previous season stats not found: {filepath}\n"
                f"Need this file to calculate qualifier averages."
            )
        
        df = pd.read_csv(filepath, encoding='latin-1')
        df = self.standardize_columns(df, player_type)
        
        logger.info(f"Loaded {len(df)} {player_type} records from {self.year - 1}")
        return df
    
    def update_player_database_from_web(self):
        """Update player ID mappings from Google Sheet."""
        
        # CSV export URL - much cleaner!
        player_id_url = 'https://docs.google.com/spreadsheets/d/1JgczhD5VDQ1EiXqVG-blttZcVwbZd5_Ne_mefUGwJnk/export?format=csv&gid=0'
        
        logger.info("Downloading latest player ID mappings...")
        print("Downloading latest player ID mappings...")
        
        # Now just read_csv - no HTML parsing needed!
        ids = pd.read_csv(player_id_url)
        
        # Clean up
        ids = ids.dropna(how='all')  # Remove empty rows
        ids = ids.drop(columns=[c for c in ids.columns if 'Unnamed' in str(c)], errors='ignore')
        
        # Get existing players
        players = pd.read_sql("SELECT * FROM players", self.engine)

        # Merge existing players with new players pulled from the Google Sheet
        # 1. Look for players in players table that have a CBSNAME and FanGraphsID but are missing a cbsid and update them
        merged_players_with_ids = players[(players['cbsid'].isna()) & players['CBSNAME'].notna()].merge(ids[['CBSNAME', 'CBSID']], on='CBSNAME', how='left', indicator=True)

        # Update the players table with the new CBSIDs where we have a match on CBSNAME and IDFANGRAPHS
        cbsids_udpated = 0
        if merged_players_with_ids.shape[0] > 0:
            with self.engine.connect() as conn:
                for i, row in merged_players_with_ids[['cbsid', 'CBSID', 'CBSNAME', 'IDFANGRAPHS']].iterrows():
                    conn.execute(text("UPDATE players SET cbsid=:cbsid WHERE CBSNAME=:CBSNAME and IDFANGRAPHS=:fgid"), {'cbsid':row['CBSID'], 'CBSNAME':row['CBSNAME'], 'fgid':row['IDFANGRAPHS']})
                    conn.commit()
                    cbsids_udpated += 1
        logger.info(f"Updated {cbsids_udpated} players missing CBSID with CBSID from Google Sheet")
        print(f"Updated {cbsids_udpated} players missing CBSID with CBSID from Google Sheet")

        # Refresh players data after updates
        players = pd.read_sql("SELECT * FROM players", self.engine)

        # 2. Look in players table for players with more than one entry by cbsid. 
        # If the player has IDFANGRAPHS for majors and minors, then move the minors version to the record of the majors and remove the minors record
        duplicate_cbsids = players[players['cbsid'].notna()].groupby('cbsid').filter(lambda x: len(x) > 1).sort_values(['cbsid', 'IDFANGRAPHS'])

        # Delete records from players table where there are duplicate cbsids and no Fangraphs information
        duplicates_removed = 0
        if duplicate_cbsids.shape[0] > 0:
            with self.engine.connect() as conn:
                for cbsid in duplicate_cbsids['cbsid'].unique():
                    result = conn.execute(text("DELETE FROM players WHERE cbsid=:cbsid AND FANGRAPHSNAME IS NULL AND IDFANGRAPHS IS NULL"), {'cbsid':cbsid})
                    duplicates_removed += result.rowcount
                    conn.commit()
        logger.info(f"Removed {duplicates_removed} duplicate player records with missing Fangraphs info")
        print(f"Removed {duplicates_removed} duplicate player records with missing Fangraphs info")
        
        # Refresh players data after updates
        players = pd.read_sql("SELECT * FROM players", self.engine)

        # 3. Finds duplicate cbsid where there are two different fangraphs IDs, moves the minors to a field in the other record then deletes the minors record
        duplicate_cbsids = players[players['cbsid'].notna()].groupby('cbsid').filter(lambda x: len(x) > 1).sort_values(['cbsid', 'IDFANGRAPHS'])
        update_count = 0
        delete_count = 0
        if duplicate_cbsids.shape[0] > 0:
            with self.engine.connect() as conn:
                for i, row in duplicate_cbsids.loc[duplicate_cbsids['IDFANGRAPHS'].str.startswith('sa'), ['cbsid', 'IDFANGRAPHS']].iterrows():
                    update_stmt = text(f"UPDATE players SET IDFANGRAPHS_minors=:fg_minors_id WHERE cbsid=:cbsid AND IDFANGRAPHS NOT LIKE 'sa%'")
                    update_result = conn.execute(update_stmt, {'fg_minors_id':row['IDFANGRAPHS'], 'cbsid':row['cbsid']})
                    update_count += update_result.rowcount
                    delete_stmt = text(f"DELETE FROM players WHERE cbsid=:cbsid AND IDFANGRAPHS_minors IS NULL AND IDFANGRAPHS LIKE 'sa%'")
                    if update_result.rowcount > 0:
                        delete_result = conn.execute(delete_stmt, {'cbsid':row['cbsid']})
                        delete_count += delete_result.rowcount
                    conn.commit()
        logger.info(f"Updated {update_count} player records with minors Fangraphs ID and removed {delete_count} duplicate minors records")
        print(f"Updated {update_count} player records with minors Fangraphs ID and removed {delete_count} duplicate minors records")

        # Refresh players data after updates
        players = pd.read_sql("SELECT * FROM players", self.engine)
        
        # 4. Removes from duplicates where YahooID is NULL
        removed_yahooid_count = 0
        duplicate_cbsids = players[players['cbsid'].notna()].groupby('cbsid').filter(lambda x: len(x) > 1).sort_values(['cbsid', 'IDFANGRAPHS'])
        if duplicate_cbsids.shape[0] > 0:
            with self.engine.connect() as conn:
                for i, row in duplicate_cbsids.iterrows():
                    if pd.isnull(row['YAHOOID']):
                        print(row['cbsid'], row['CBSNAME'], row['YAHOOID'])
                        result = conn.execute(text("DELETE FROM players WHERE YAHOOID IS NULL AND cbsid=:cbsid AND CBSNAME=:CBSNAME"), {'cbsid':row['cbsid'], 'CBSNAME':row['CBSNAME']})
                        conn.commit()
                        removed_yahooid_count += result.rowcount
        logger.info(f"Removed {removed_yahooid_count} duplicate player records with missing YahooID")
        print(f"Removed {removed_yahooid_count} duplicate player records with missing YahooID")

        # Refresh players data after updates
        players = pd.read_sql("SELECT * FROM players", self.engine)

        # 5. Add records from Google sheet to players table if they don't already exist
        insert_count = 0
        merged = players.merge(ids[ids['CBSID'].notna()][['CBSNAME', 'CBSID', 'IDFANGRAPHS']], left_on='cbsid', right_on='CBSID', how='outer', indicator=True)
        if merged._merge.value_counts()['right_only'] > 0:
            with self.engine.connect() as conn:
                for i, row in ids[ids['CBSID'].isin(merged[merged['_merge']=='right_only']['CBSID'].tolist())].iterrows():                    
                    for key, val in row.to_dict().items():
                        if pd.isnull(val):
                            row[key] = None
                    row['IDFANGRAPHS_minors'] = None
                    insert_stmt = text("INSERT INTO players VALUES (:CBSID, :CBSNAME, :IDPLAYER, :PLAYERNAME, :BIRTHDATE, :FIRSTNAME, :LASTNAME, :TEAM, :LG, :POS, :IDFANGRAPHS, :FANGRAPHSNAME, :MLBID, :MLBNAME, :RETROID, :BREFID, :NFBCID, \
                            :NFBCNAME, :ESPNID, :ESPNNAME, :BPID, :YAHOOID, :YAHOONAME, :MSTRBLLNAME, :BATS, :THROWS, :FANTPROSNAME, :LASTCOMMAFIRST, :ROTOWIREID, :FANTRAXID, :FANTRAXNAME, :ROTOWIRENAME, :ALLPOS, :NFBCLASTFIRST, :ACTIVE, :IDFANGRAPHS_minors)")
                    result = conn.execute(insert_stmt, row[[key for key in row.to_dict().keys() if key.lower() in players.columns or key in players.columns]].to_dict())
                    conn.commit()
                    insert_count += result.rowcount
        logger.info(f"Inserted {insert_count} new player records from Google Sheet")
        print(f"Inserted {insert_count} new player records from Google Sheet")
        
        return ids
    
    # ========================================================================
    # Column Handling (Dynamic, No Hardcoding)
    # ========================================================================
    
    def standardize_columns(
        self, 
        df: pd.DataFrame, 
        player_type: Literal['hitting', 'pitching']
    ) -> pd.DataFrame:
        """
        Intelligently standardize column names from different data sources.
        
        Args:
            df: DataFrame with potentially non-standard columns
            player_type: 'hitting' or 'pitching'
            
        Returns:
            DataFrame with standardized column names
        """
        rename_map = {}
        
        # Handle common variations
        for standard, variations in self.column_mappings.items():
            for variant in variations:
                if variant in df.columns and variant != standard:
                    rename_map[variant] = standard
                    break
        
        # Handle player type specific columns
        if player_type == 'pitching' and 'H' in df.columns:
            # For pitchers, H should be HA (hits allowed)
            rename_map['H'] = 'HA'
        
        # Standard name fixes
        common_renames = {
            'NameASCII': 'Name',
            'PlayerId': 'playerid',
            'Positions': 'Pos',
        }
        
        for old, new in common_renames.items():
            if old in df.columns and new not in df.columns:
                rename_map[old] = new
        
        # Drop junk columns
        junk_cols = ['ï»¿Name', 'Unnamed: 0']
        df = df.drop(columns=[c for c in junk_cols if c in df.columns], errors='ignore')
        
        if rename_map:
            df = df.rename(columns=rename_map)
            logger.debug(f"Standardized columns: {rename_map}")
        
        return df
    
    def average_projections(
        self, 
        df: pd.DataFrame, 
        player_type: Literal['hitting', 'pitching'],
        id_col: str = 'cbsid'
    ) -> pd.DataFrame:
        """
        Average projections across all systems dynamically.
        
        No hardcoded stat lists - automatically finds numeric columns to average.
        
        Args:
            df: Combined projections from multiple systems
            player_type: 'hitting' or 'pitching'
            
        Returns:
            DataFrame with averaged projections
        """
        # Columns that identify the player (don't average)
        id_cols = ['cbsid', 'playerid', 'Name', 'Team', 'Pos']
        
        # Find all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Stats to average = numeric columns that aren't IDs
        stat_cols = [col for col in numeric_cols if col not in id_cols]
        
        logger.info(f"Averaging {len(stat_cols)} stats across projection systems")
        logger.debug(f"Stats being averaged: {stat_cols}")
        
        # Average the stats by player
        averaged = (df.groupby(id_col, as_index=False)[stat_cols]
                    .mean())
        
        # Get player info (take first occurrence)
        player_info = (df.groupby(id_col, as_index=False)[id_cols]
                       .first())
        
        result = averaged.merge(player_info, on=id_col, how='left')
        
        return result
    
    def add_last_year_suffix(
        self, 
        df: pd.DataFrame,
        exclude_cols: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Add '_ly' suffix to stat columns for merging with projections.
        
        Dynamic - works with any stats present in the DataFrame.
        
        Args:
            df: DataFrame with previous season stats
            exclude_cols: Columns to not suffix (defaults to common IDs)
            
        Returns:
            DataFrame with '_ly' suffixed stat columns
        """
        if exclude_cols is None:
            exclude_cols = ['cbsid', 'playerid', 'Name', 'Team', 'Pos', 'MLBID']
        
        rename_dict = {
            col: f"{col}_ly"
            for col in df.columns
            if col not in exclude_cols
        }
        
        return df.rename(columns=rename_dict)
    
    # ========================================================================
    # Projection Processing
    # ========================================================================
    
    def process_hitting_projections(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process hitting projections: calculate rates, add position info, merge external data.
        
        Args:
            df: Averaged hitting projections
            
        Returns:
            Processed hitting projections
        """
        # Calculate derived stats if base stats exist
        if all(col in df.columns for col in ['H', 'AB']):
            df['BA'] = df['H'] / df['AB']
        
        if all(col in df.columns for col in ['H', 'BB', 'HBP', 'AB', 'SF']):
            df['OBP'] = (df['H'] + df['BB'] + df['HBP']) / (df['AB'] + df['BB'] + df['HBP'] + df['SF'])
        
        if all(col in df.columns for col in ['TB', 'AB']):
            df['SLG'] = df['TB'] / df['AB']
        
        if 'OBP' in df.columns and 'SLG' in df.columns:
            df['OPS'] = df['OBP'] + df['SLG']
        
        if all(col in df.columns for col in ['K', 'PA']):
            df['K%'] = df['K'] / df['PA']
        
        if all(col in df.columns for col in ['BB', 'PA']):
            df['BB%'] = df['BB'] / df['PA']
        
        # Determine primary position
        if 'Pos' in df.columns:
            df['Primary_Pos'] = df['Pos'].apply(self.find_primary_position)
        
        # Create a sorting value (higher = better)
        df['_sort_value'] = 0
        for stat in ['HR', 'R', 'RBI', 'SB', 'H']:
            if stat in df.columns:
                df['_sort_value'] += df[stat].fillna(0)
        
        # Try to merge with external data sources (CBS, FanGraphs, etc.)
        df = self.merge_external_data(df, 'hitting')
        
        # Sort by projected value
        df = df.sort_values('_sort_value', ascending=False)
        
        return df
    
    def process_pitching_projections(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process pitching projections: calculate rates, add position info, merge external data.
        
        Args:
            df: Averaged pitching projections
            
        Returns:
            Processed pitching projections
        """
        # Calculate derived stats
        if all(col in df.columns for col in ['ER', 'IP']):
            df['ERA'] = (df['ER'] / df['IP']) * 9
        
        if all(col in df.columns for col in ['HA', 'BB', 'IP']):
            df['WHIP'] = (df['HA'] + df['BB']) / df['IP']
        
        if all(col in df.columns for col in ['SO', 'IP']):
            df['K/9'] = (df['SO'] / df['IP']) * 9
        
        if all(col in df.columns for col in ['SO', 'TBF']):
            df['K%'] = df['SO'] / df['TBF']
        
        if all(col in df.columns for col in ['BB', 'TBF']):
            df['BB%'] = df['BB'] / df['TBF']
            df['K-BB%'] = df['K%'] - df['BB%']
        
        # Calculate SvHld if components exist
        if 'SvHld' not in df.columns and all(col in df.columns for col in ['SV', 'HLD']):
            df['SvHld'] = df['SV'] + df['HLD']
        
        # Determine position (SP vs RP)
        if 'Pos' in df.columns:
            df['Primary_Pos'] = df['Pos'].apply(
                lambda x: self.find_primary_position(x) if pd.notna(x) else 'RP'
            )
        elif 'GS' in df.columns:
            # Guess based on starts
            df['Primary_Pos'] = df['GS'].apply(lambda gs: 'SP' if gs > 0 else 'RP')
        
        # Create a sorting value
        df['_sort_value'] = 0
        for stat, weight in [('SO', 1), ('SvHld', 4), ('QS', 1)]:
            if stat in df.columns:
                df['_sort_value'] += df[stat].fillna(0) * weight
        
        # Try to merge with external data
        df = self.merge_external_data(df, 'pitching')
        
        df = df.sort_values('_sort_value', ascending=False)
        
        return df
    
    def merge_external_data(
        self, 
        df: pd.DataFrame, 
        player_type: Literal['hitting', 'pitching']
    ) -> pd.DataFrame:
        """
        Merge with external data sources (CBS values, FanGraphs, etc.).
        
        Fails gracefully if files don't exist.
        
        Args:
            df: Projection DataFrame
            player_type: 'hitting' or 'pitching'
            
        Returns:
            DataFrame with external data merged (if available)
        """
        suffix = 'h' if player_type == 'hitting' else 'p'
        
        # Try CBS auction values
        cbs_path = self.data_dir / f"{self.year}-cbs-auction-values.csv"
        if cbs_path.exists():
            try:
                cbs = pd.read_csv(cbs_path)[['cbsid', 'CBS']].drop_duplicates('cbsid')
                df = df.merge(cbs, on='cbsid', how='left')
                df['CBS'] = df['CBS'].fillna(0)
                logger.info("Merged CBS auction values")
            except Exception as e:
                logger.warning(f"Could not merge CBS values: {e}")
                print(f"Could not merge CBS values: {e}")
        
        # Try FanGraphs auction calculator
        fg_path = self.data_dir / f"{self.year}-fangraphs-auction-calculator-{suffix}.csv"
        if fg_path.exists():
            try:
                fg = pd.read_csv(fg_path)[['PlayerId', 'Dollars']].drop_duplicates('PlayerId')
                df = df.merge(fg, left_on='playerid', right_on='PlayerId', how='left')
                df = df.drop(columns=['PlayerId'], errors='ignore')
                df['Dollars'] = df['Dollars'].fillna(0)
                logger.info("Merged FanGraphs auction values")
            except Exception as e:
                logger.warning(f"Could not merge FanGraphs values: {e}")
                print(f"Could not merge FanGraphs values: {e}")
        
        # Try previous season stats for comparison
        prev_path = self.data_dir / f"{self.year - 1}-final-stats-{suffix}.csv"
        if prev_path.exists():
            try:
                prev = pd.read_csv(prev_path, encoding='latin-1')
                prev = self.standardize_columns(prev, player_type)
                prev = self.add_last_year_suffix(prev)
                
                # Only merge columns that don't already exist
                merge_cols = ['cbsid'] + [c for c in prev.columns if c.endswith('_ly')]
                df = df.merge(prev[merge_cols], on='cbsid', how='left')
                logger.info(f"Merged {self.year - 1} stats")
            except Exception as e:
                logger.warning(f"Could not merge previous season stats: {e}")
                print(f"Could not merge previous season stats: {e}")
        
        return df
    
    # ========================================================================
    # Qualifier and Z-Score Calculation
    # ========================================================================
    
    def calculate_qualifiers(
        self,
        hitting: pd.DataFrame,
        pitching: pd.DataFrame,
        min_pa: int = 440,
        min_ip: float = 130,
        sp_min_ip: float = 130,
        rp_ip_range: list = [45, 95],
        min_sv_hld: int = 5,
        hitting_percentile: float = 0.65,
        pitching_percentile: float = 0.65
    ) -> dict:
        """
        Calculate league qualifier averages dynamically.
        
        Instead of hardcoded PA minimums, uses percentiles of playing time.
        
        Args:
            hitting: Previous season hitting stats
            pitching: Previous season pitching stats
            hitting_percentile: What percentile qualifies for hitters
            pitching_percentile: What percentile qualifies for pitchers
            
        Returns:
            Dictionary with mean and std for each stat category
        """
        qualifiers = {}
        
        # Hitting qualifiers
        if 'PA' in hitting.columns:
            pa_cutoff = hitting[hitting['PA'] > 0]['PA'].quantile(hitting_percentile)
            qual_hitters = hitting[hitting['PA'] >= pa_cutoff].copy()
            logger.info(f"Found {len(qual_hitters)} qualifying hitters (PA >= {pa_cutoff:.0f})")
            
            # Calculate league batting average
            lg_ba = qual_hitters['H'].sum() / qual_hitters['AB'].sum()
            
            # Calculate zlgBA (league-adjusted batting average)
            qual_hitters['zlgBA'] = qual_hitters['H'] - (qual_hitters['AB'] * lg_ba)
            
            # Get mean and std for counting stats
            counting_stats = ['H', 'HR', 'R', 'RBI', 'SB', 'AB', 'PA', 'zlgBA']
            for stat in counting_stats:
                if stat in qual_hitters.columns:
                    qualifiers[stat] = {
                        'mean': qual_hitters[stat].mean(),
                        'std': qual_hitters[stat].std()
                    }
            
            # Add BA separately
            qualifiers['BA'] = {
                'mean': lg_ba,
                'std': (qual_hitters['H'] / qual_hitters['AB']).std()
            }
        
        # Pitching qualifiers
        if 'IP' in pitching.columns:
            # Separate SP and RP
            if 'Primary_Pos' in pitching.columns:
                sp_cutoff = pitching[
                    (pitching['Primary_Pos'] == 'SP') & (pitching['IP'] > 0)
                ]['IP'].quantile(pitching_percentile)
                
                rp_cutoff = pitching[
                    (pitching['Primary_Pos'] == 'RP') & (pitching['IP'] > 0)
                ]['IP'].quantile(pitching_percentile)  # Lower bar for relievers
                
                qual_pitchers = pitching[
                    ((pitching['Primary_Pos'] == 'SP') & (pitching['IP'] >= sp_cutoff)) |
                    ((pitching['Primary_Pos'] == 'RP') & (pitching['IP'].between(45,95)) & (pitching.get('SvHld', 0) > 5))
                ].copy()
            else:
                ip_cutoff = pitching[pitching['IP'] > 0]['IP'].quantile(pitching_percentile)
                qual_pitchers = pitching[pitching['IP'] >= ip_cutoff].copy()
            
            logger.info(f"Found {len(qual_pitchers)} qualifying pitchers")
            print(f"Found {len(qual_pitchers)} qualifying pitchers")
            
            # Calculate league ERA and WHIP
            lg_era = (qual_pitchers['ER'].sum() / qual_pitchers['IP'].sum()) * 9
            lg_whip = (qual_pitchers['HA'].sum() + qual_pitchers['BB'].sum()) / qual_pitchers['IP'].sum()
            
            # Calculate zlg stats (league-adjusted)
            qual_pitchers['zlgERA'] = ((qual_pitchers['ER'] * 9) - (qual_pitchers['IP'] * lg_era)) * -1
            qual_pitchers['zlgWHIP'] = ((qual_pitchers['HA'] + qual_pitchers['BB']) - (qual_pitchers['IP'] * lg_whip)) * -1
            
            # Get mean and std for counting stats
            counting_stats = ['BB', 'HA', 'ER', 'IP', 'SO', 'QS', 'SvHld', 'zlgERA', 'zlgWHIP']
            for stat in counting_stats:
                if stat in qual_pitchers.columns:
                    qualifiers[stat] = {
                        'mean': qual_pitchers[stat].mean(),
                        'std': qual_pitchers[stat].std()
                    }
            
            # Add ERA and WHIP separately
            qualifiers['ERA'] = {
                'mean': lg_era,
                'std': ((qual_pitchers['ER'] / qual_pitchers['IP']) * 9).std()
            }
            qualifiers['WHIP'] = {
                'mean': lg_whip,
                'std': ((qual_pitchers['HA'] + qual_pitchers['BB']) / qual_pitchers['IP']).std()
            }
        
        logger.info(f"Calculated qualifiers for {len(qualifiers)} stat categories")
        return qualifiers
    
    def calculate_z_score(
        self, 
        value: float, 
        stat: str, 
        qualifiers: dict,
        inverse: bool = False
    ) -> float:
        """
        Calculate z-score for a single value.
        
        Args:
            value: The stat value
            stat: Stat name
            qualifiers: Qualifier dictionary
            inverse: If True, lower is better (ERA, WHIP)
            
        Returns:
            Z-score
        """
        if stat not in qualifiers:
            logger.warning(f"No qualifier data for {stat}, returning 0")
            return 0.0
        
        mean = qualifiers[stat]['mean']
        std = qualifiers[stat]['std']
        
        if std == 0:
            return 0.0
        
        z = (value - mean) / std
        
        if inverse:
            z = -z
        
        return z
    
    def calculate_z_scores(
        self, 
        df: pd.DataFrame, 
        player_type: Literal['hitting', 'pitching']
    ) -> pd.DataFrame:
        """
        Calculate z-scores for all stats in the DataFrame.
        
        Dynamically handles whatever stats are present.
        
        Args:
            df: Projection DataFrame
            player_type: 'hitting' or 'pitching'
            
        Returns:
            DataFrame with z-score columns added
        """
        if self.qualifiers is None:
            raise ValueError("Must calculate qualifiers first")
        
        df = df.copy()
        
        if player_type == 'hitting':
            # Define stats and whether they're inverse
            stats = {
                'R': False,
                'HR': False,
                'RBI': False,
                'SB': False,
            }
            
            # Handle BA specially (rate stat)
            if all(col in df.columns for col in ['H', 'AB']):
                df['zBA'] = df.apply(
                    lambda row: self._calculate_ba_zscore(row),
                    axis=1
                )
            
            # Calculate z-scores for counting stats
            for stat, inverse in stats.items():
                if stat in df.columns:
                    df[f'z{stat}'] = df[stat].apply(
                        lambda x: self.calculate_z_score(x, stat, self.qualifiers, inverse)
                    )
            
            # Aggregate z-score (BIGAA = sum of z-scores)
            z_cols = [c for c in df.columns if c.startswith('z') and c != 'zlgBA']
            if z_cols:
                df['total_z'] = df[z_cols].sum(axis=1)
        
        else:  # pitching
            stats = {
                'SO': False,
                'QS': False,
                'SvHld': False,
            }
            
            # Handle ERA and WHIP specially (rate stats, inverse)
            if all(col in df.columns for col in ['ER', 'IP']):
                df['zERA'] = df.apply(
                    lambda row: self._calculate_era_zscore(row),
                    axis=1
                )
            
            if all(col in df.columns for col in ['HA', 'BB', 'IP']):
                df['zWHIP'] = df.apply(
                    lambda row: self._calculate_whip_zscore(row),
                    axis=1
                )
            
            # Calculate z-scores for counting stats
            for stat, inverse in stats.items():
                if stat in df.columns:
                    df[f'z{stat}'] = df[stat].apply(
                        lambda x: self.calculate_z_score(x, stat, self.qualifiers, inverse)
                    )
            
            # Aggregate z-score
            z_cols = [c for c in df.columns if c.startswith('z') and c not in ['zlgERA', 'zlgWHIP']]
            if z_cols:
                df['total_z'] = df[z_cols].sum(axis=1)
        
        return df
    
    def _calculate_ba_zscore(self, row: pd.Series) -> float:
        """Calculate z-score for batting average (rate stat)."""
        if pd.isna(row['H']) or pd.isna(row['AB']) or row['AB'] == 0:
            return 0.0
        
        quals = self.qualifiers
        ba_points = row['H'] - (row['AB'] * (quals['H']['mean'] / quals['AB']['mean']))
        return (ba_points - quals['zlgBA']['mean']) / quals['zlgBA']['std']
    
    def _calculate_era_zscore(self, row: pd.Series) -> float:
        """Calculate z-score for ERA (rate stat, inverse)."""
        if pd.isna(row['ER']) or pd.isna(row['IP']) or row['IP'] == 0:
            return 0.0
        
        quals = self.qualifiers
        era_points = ((row['ER'] * 9) - (row['IP'] * quals['ERA']['mean'])) * -1
        return (era_points - quals['zlgERA']['mean']) / quals['zlgERA']['std']
    
    def _calculate_whip_zscore(self, row: pd.Series) -> float:
        """Calculate z-score for WHIP (rate stat, inverse)."""
        if pd.isna(row['IP']) or row['IP'] == 0:
            return 0.0
        
        quals = self.qualifiers
        whip_points = ((row['HA'] + row['BB']) - (row['IP'] * quals['WHIP']['mean'])) * -1
        return (whip_points - quals['zlgWHIP']['mean']) / quals['zlgWHIP']['std']
    
    # ========================================================================
    # Positional Adjustments and Value Calculation
    # ========================================================================
    
    def apply_positional_adjustments(
        self, 
        df: pd.DataFrame, 
        player_type: Literal['hitting', 'pitching']
    ) -> pd.DataFrame:
        """
        Apply positional scarcity adjustments to z-scores.
        
        Args:
            df: DataFrame with z-scores
            player_type: 'hitting' or 'pitching'
            
        Returns:
            DataFrame with positional adjustments applied
        """
        df = df.copy()
        
        if 'total_z' not in df.columns:
            logger.warning("No total_z column found, skipping positional adjustments")
            return df
        
        # Calculate replacement level for each position
        pos_adjustments = self._calculate_replacement_levels(df, player_type)
        
        # Apply adjustment based on player's positions
        if 'Pos' in df.columns:
            df['pos_adj'] = df['Pos'].apply(
                lambda pos_str: self._get_position_adjustment(pos_str, pos_adjustments)
            )
        else:
            df['pos_adj'] = 0
        
        # Final z-score = base z + positional adjustment
        df['z'] = df['total_z'] + df['pos_adj']
        
        return df
    
    def _calculate_replacement_levels(
        self, 
        df: pd.DataFrame, 
        player_type: Literal['hitting', 'pitching']
    ) -> dict:
        """
        Calculate replacement level (last drafted player) for each position.
        
        Returns:
            Dictionary of {position: adjustment_value}
        """
        adjustments = {}
        
        if player_type == 'hitting':
            positions = ['C', '1B', '2B', '3B', 'SS', 'OF']
        else:
            positions = ['SP', 'RP']
        
        for pos in positions:
            # Find players eligible at this position
            eligible = df[df['Primary_Pos'] == pos].copy()
            
            if len(eligible) == 0:
                adjustments[pos] = 0
                continue
            
            # Find the replacement player (last one drafted)
            num_drafted = int(self.n_teams * self.position_requirements.get(pos, 1))
            
            if len(eligible) < num_drafted:
                logger.warning(
                    f"Only {len(eligible)} {pos} players available, need {num_drafted}"                        
                )
                num_drafted = len(eligible)
            
            # Replacement level = z-score of last drafted player
            eligible_sorted = eligible.sort_values('total_z', ascending=False)
            replacement_z = eligible_sorted.iloc[num_drafted - 1]['total_z']
            
            # Adjustment is the abs value of replacement level
            adjustments[pos] = replacement_z * -1
            
            logger.debug(f"{pos}: {num_drafted} drafted, replacement z={replacement_z:.2f}")
            print(f"{pos}: {num_drafted} drafted, replacement z={replacement_z:.2f}, adjustment={adjustments[pos]:.2f}")
        
        return adjustments
    
    def _get_position_adjustment(self, pos_string: str, adjustments: dict) -> float:
        """
        Get the maximum position adjustment for a player.
        
        Players get credit for their most valuable position.
        """
        if pd.isna(pos_string):
            return 0.0
        
        # Parse positions
        positions = re.split('[,/]', str(pos_string))
        
        # Get max adjustment across all eligible positions
        eligible_adjustments = [
            adjustments.get(pos, 0) 
            for pos in positions 
            if pos in adjustments
        ]
        
        return max(eligible_adjustments) if eligible_adjustments else 0.0
    
    def calculate_auction_values(
        self, 
        hitting: pd.DataFrame, 
        pitching: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert z-scores to auction values.
        
        Args:
            hitting: Hitting projections with z-scores
            pitching: Pitching projections with z-scores
            
        Returns:
            Combined DataFrame with auction values
        """
        # Calculate conversion factor ($/z-score)
        total_positive_z = (
            hitting[hitting['z'] > 0]['z'].sum() +
            pitching[pitching['z'] > 0]['z'].sum()
        )
        
        conversion_factor = (
            (self.budget_per_team / self.roster_size) *
            (self.total_players / total_positive_z)
        )
        
        logger.info(f"Conversion factor: ${conversion_factor:.2f} per z-score")
        print(f"Conversion factor: ${conversion_factor:.2f} per z-score")
        logger.info(f"Total positive z-scores: {total_positive_z:.1f}")
        print(f"Total positive z-scores: {total_positive_z:.1f}")
        
        # Calculate values
        hitting['Value'] = hitting['z'] * conversion_factor
        pitching['Value'] = pitching['z'] * conversion_factor
        
        # Combine
        combined = pd.concat([hitting, pitching], ignore_index=True)
        
        # Clean up negative values
        #combined['Value'] = combined['Value'].clip(lower=0)
        
        # Round to nearest dollar
        combined['Value'] = combined['Value'].round(1).astype(float)
        
        # Sort by value
        combined = combined.sort_values('Value', ascending=False)
        
        return combined
    

    # ========================================================================
    # SHARED CALCULATION METHODS
    # ========================================================================
    
    def _convert_to_dollars(
        self,
        hitting: pd.DataFrame,
        pitching: pd.DataFrame,
        z_column: str = 'z',
        normalize_to_budget: bool = True
    ) -> pd.DataFrame:
        """
        Convert z-scores to dollar values.
        
        Args:
            hitting: Hitting DataFrame with z-scores
            pitching: Pitching DataFrame with z-scores
            z_column: Which z-score column to use ('z' or 'total_z')
            normalize_to_budget: Scale to league budget
            
        Returns:
            Combined DataFrame with dollar values
        """
        # Calculate total positive z-scores
        total_positive_z = (
            hitting[hitting[z_column] > 0][z_column].sum() +
            pitching[pitching[z_column] > 0][z_column].sum()
        )
        
        # Calculate conversion factor
        if normalize_to_budget:
            conversion_factor = (
                (self.budget_per_team / self.roster_size) *
                (self.total_players / total_positive_z)
            )
        else:
            # Use draft conversion factor if available
            conversion_factor = self.draft_conversion_factor or 5.0
        
        # Store for reference
        self._last_conversion_factor = conversion_factor
        
        logger.info(f"Total positive z-scores: {total_positive_z:.1f}")
        logger.info(f"Conversion factor: ${conversion_factor:.2f} per z-score")
        
        # Calculate values
        hitting['Value'] = hitting[z_column] * conversion_factor
        pitching['Value'] = pitching[z_column] * conversion_factor
        
        # Combine
        combined = pd.concat([hitting, pitching], ignore_index=True)
        
        # Round values
        combined['Value'] = combined['Value'].round(1).astype(float)
        
        # Sort by value
        combined = combined.sort_values('Value', ascending=False)
        
        return combined
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def find_primary_position(self, pos_string: str) -> Optional[str]:
        """
        Determine primary position from position string.
        
        Args:
            pos_string: e.g., "2B,SS,OF" or "SP/RP"
            
        Returns:
            Primary position based on scarcity hierarchy
        """
        if pd.isna(pos_string):
            return None
        
        # Parse positions
        positions = re.split('[,/]', str(pos_string).strip())
        
        # Check hierarchy in order of scarcity
        for pos in self.position_hierarchy:
            if pos in positions:
                return pos
        
        # Fallback
        return positions[0] if positions else None
    
    def load_player_index(self) -> pd.DataFrame:
        """
        Docstring for load_player_index
        
        :param self: Description
        :return: Description
        :rtype: DataFrame
        """
        # Load player index from database
        with self.engine.connect() as conn:
            players = pd.read_sql("SELECT * FROM players", conn)
            self.player_index = players
        
        logger.info(f"Loaded player index with {len(players)} records")
        return players
    
    def save_to_database(
        self, 
        table_name: str = 'projections',
        db_path: Optional[str] = None
    ):
        """
        Save final projections to SQLite database.
        
        Args:
            table_name: Name of table to create
            db_path: Path to database (defaults to data_dir/fantasy_data.db)
        """
        if self.final_values is None:
            raise ValueError("No data to save. Run generate_auction_values() first.")
        
        if db_path is None:
            db_path = self.data_dir / 'fantasy_data.db'
        
        engine = create_engine(f'sqlite:///{db_path}')
        
        self.final_values.to_sql(
            f'{table_name}{self.year}',
            engine,
            if_exists='replace',
            index=False
        )
        
        logger.info(f"Saved {len(self.final_values)} rows to {table_name}{self.year}")
    
    def get_top_players(self, n: int = 20, position: Optional[str] = None) -> pd.DataFrame:
        """
        Get top players by value.
        
        Args:
            n: Number of players to return
            position: Filter to specific position (optional)
            
        Returns:
            DataFrame with top players
        """
        if self.final_values is None:
            raise ValueError("No data available. Run generate_auction_values() first.")
        
        df = self.final_values
        
        if position:
            df = df[df['Pos'].str.contains(position, na=False)]
        
        return df.head(n)[['Name', 'Team', 'Pos', 'Value', 'z', 'total_z']]


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize with your league settings
    fp = FantasyProjections(
        data_dir="data",
        year=2025,
        n_teams=12,
        roster_size=23,
        budget_per_team=260,
        hitter_budget_split=0.6
    )
    
    # Generate values (auto-discovers projection systems)
    values = fp.generate_auction_values()
    
    # View top players
    print("\nTop 20 Players:")
    print(fp.get_top_players(20))
    
    print("\nTop 10 Catchers:")
    print(fp.get_top_players(10, position='C'))
    
    # Save to database
    fp.save_to_database('projections')


