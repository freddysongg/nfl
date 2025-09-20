"""
NFL Data Pipeline Schema Definitions
Updated schema definitions based on actual 2025 NFL data structure
Generated from real data investigation on 2025-09-20
"""

import duckdb


def create_raw_player_stats_table(conn: duckdb.DuckDBPyConnection):
    """Create raw_player_stats table matching actual 2025 data (114 columns)"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_player_stats (
            -- Basic player info (11 columns)
            player_id VARCHAR,
            player_name VARCHAR,
            player_display_name VARCHAR,
            position VARCHAR,
            position_group VARCHAR,
            headshot_url VARCHAR,
            season INTEGER,
            week INTEGER,
            season_type VARCHAR,
            team VARCHAR,
            opponent_team VARCHAR,

            -- Passing stats (16 columns)
            completions INTEGER,
            attempts INTEGER,
            passing_yards INTEGER,
            passing_tds INTEGER,
            passing_interceptions INTEGER,
            sacks_suffered INTEGER,
            sack_yards_lost INTEGER,
            sack_fumbles INTEGER,
            sack_fumbles_lost INTEGER,
            passing_air_yards INTEGER,
            passing_yards_after_catch INTEGER,
            passing_first_downs INTEGER,
            passing_epa FLOAT,
            passing_cpoe FLOAT,
            passing_2pt_conversions INTEGER,
            pacr FLOAT,

            -- Rushing stats (8 columns)
            carries INTEGER,
            rushing_yards INTEGER,
            rushing_tds INTEGER,
            rushing_fumbles INTEGER,
            rushing_fumbles_lost INTEGER,
            rushing_first_downs INTEGER,
            rushing_epa FLOAT,
            rushing_2pt_conversions INTEGER,

            -- Receiving stats (15 columns)
            receptions INTEGER,
            targets INTEGER,
            receiving_yards INTEGER,
            receiving_tds INTEGER,
            receiving_fumbles INTEGER,
            receiving_fumbles_lost INTEGER,
            receiving_air_yards INTEGER,
            receiving_yards_after_catch INTEGER,
            receiving_first_downs INTEGER,
            receiving_epa FLOAT,
            receiving_2pt_conversions INTEGER,
            racr FLOAT,
            target_share FLOAT,
            air_yards_share FLOAT,
            wopr FLOAT,

            -- Special teams (1 column)
            special_teams_tds INTEGER,

            -- Defensive stats (15 columns)
            def_tackles_solo INTEGER,
            def_tackles_with_assist INTEGER,
            def_tackle_assists INTEGER,
            def_tackles_for_loss INTEGER,
            def_tackles_for_loss_yards INTEGER,
            def_fumbles_forced INTEGER,
            def_sacks INTEGER,
            def_sack_yards FLOAT,
            def_qb_hits INTEGER,
            def_interceptions INTEGER,
            def_interception_yards INTEGER,
            def_pass_defended INTEGER,
            def_tds INTEGER,
            def_fumbles INTEGER,
            def_safeties INTEGER,

            -- Miscellaneous (19 columns)
            misc_yards INTEGER,
            fumble_recovery_own INTEGER,
            fumble_recovery_yards_own INTEGER,
            fumble_recovery_opp INTEGER,
            fumble_recovery_yards_opp INTEGER,
            fumble_recovery_tds INTEGER,
            penalties INTEGER,
            penalty_yards INTEGER,
            punt_returns INTEGER,
            punt_return_yards INTEGER,
            punt_return_tds INTEGER,
            kickoff_returns INTEGER,
            kickoff_return_yards INTEGER,
            kickoff_return_tds INTEGER,
            fg_att INTEGER,
            fg_made INTEGER,
            fg_missed INTEGER,
            fg_blocked INTEGER,
            fg_long INTEGER,
            fg_pct FLOAT,
            fg_made_0_19 INTEGER,
            fg_made_20_29 INTEGER,
            fg_made_30_39 INTEGER,
            fg_made_40_49 INTEGER,
            fg_made_50_59 INTEGER,
            fg_made_60_ INTEGER,
            fg_missed_0_19 INTEGER,
            fg_missed_20_29 INTEGER,
            fg_missed_30_39 INTEGER,
            fg_missed_40_49 INTEGER,
            fg_missed_50_59 INTEGER,
            fg_missed_60_ INTEGER,
            pat_att INTEGER,
            pat_made INTEGER,
            pat_missed INTEGER,
            pat_blocked INTEGER,
            pat_pct FLOAT,
            gwfg_att INTEGER,
            gwfg_made INTEGER,
            gwfg_missed INTEGER,
            gwfg_blocked INTEGER,
            fantasy_points FLOAT,
            fantasy_points_ppr FLOAT
        )
    """)


def create_raw_team_stats_table(conn: duckdb.DuckDBPyConnection):
    """Create raw_team_stats table matching actual 2025 data (102 columns)"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_team_stats (
            -- Basic team info (5 columns)
            season INTEGER,
            week INTEGER,
            team VARCHAR,
            season_type VARCHAR,
            opponent_team VARCHAR,

            -- Passing offense (16 columns)
            completions INTEGER,
            attempts INTEGER,
            passing_yards INTEGER,
            passing_tds INTEGER,
            passing_interceptions INTEGER,
            sacks_suffered INTEGER,
            sack_yards_lost INTEGER,
            sack_fumbles INTEGER,
            sack_fumbles_lost INTEGER,
            passing_air_yards INTEGER,
            passing_yards_after_catch INTEGER,
            passing_first_downs INTEGER,
            passing_epa FLOAT,
            passing_cpoe FLOAT,
            passing_2pt_conversions INTEGER,
            pacr FLOAT,

            -- Rushing offense (8 columns)
            carries INTEGER,
            rushing_yards INTEGER,
            rushing_tds INTEGER,
            rushing_fumbles INTEGER,
            rushing_fumbles_lost INTEGER,
            rushing_first_downs INTEGER,
            rushing_epa FLOAT,
            rushing_2pt_conversions INTEGER,

            -- Receiving offense (15 columns)
            receptions INTEGER,
            targets INTEGER,
            receiving_yards INTEGER,
            receiving_tds INTEGER,
            receiving_fumbles INTEGER,
            receiving_fumbles_lost INTEGER,
            receiving_air_yards INTEGER,
            receiving_yards_after_catch INTEGER,
            receiving_first_downs INTEGER,
            receiving_epa FLOAT,
            receiving_2pt_conversions INTEGER,
            racr FLOAT,
            target_share FLOAT,
            air_yards_share FLOAT,
            wopr FLOAT,

            -- Special teams (1 column)
            special_teams_tds INTEGER,

            -- Defensive stats (15 columns)
            def_tackles_solo INTEGER,
            def_tackles_with_assist INTEGER,
            def_tackle_assists INTEGER,
            def_tackles_for_loss INTEGER,
            def_tackles_for_loss_yards INTEGER,
            def_fumbles_forced INTEGER,
            def_sacks INTEGER,
            def_sack_yards FLOAT,
            def_qb_hits INTEGER,
            def_interceptions INTEGER,
            def_interception_yards INTEGER,
            def_pass_defended INTEGER,
            def_tds INTEGER,
            def_fumbles INTEGER,
            def_safeties INTEGER,

            -- Miscellaneous (42 columns)
            misc_yards INTEGER,
            fumble_recovery_own INTEGER,
            fumble_recovery_yards_own INTEGER,
            fumble_recovery_opp INTEGER,
            fumble_recovery_yards_opp INTEGER,
            fumble_recovery_tds INTEGER,
            penalties INTEGER,
            penalty_yards INTEGER,
            punt_returns INTEGER,
            punt_return_yards INTEGER,
            punt_return_tds INTEGER,
            kickoff_returns INTEGER,
            kickoff_return_yards INTEGER,
            kickoff_return_tds INTEGER,
            fg_att INTEGER,
            fg_made INTEGER,
            fg_missed INTEGER,
            fg_blocked INTEGER,
            fg_long INTEGER,
            fg_pct FLOAT,
            fg_made_0_19 INTEGER,
            fg_made_20_29 INTEGER,
            fg_made_30_39 INTEGER,
            fg_made_40_49 INTEGER,
            fg_made_50_59 INTEGER,
            fg_made_60_ INTEGER,
            fg_missed_0_19 INTEGER,
            fg_missed_20_29 INTEGER,
            fg_missed_30_39 INTEGER,
            fg_missed_40_49 INTEGER,
            fg_missed_50_59 INTEGER,
            fg_missed_60_ INTEGER,
            pat_att INTEGER,
            pat_made INTEGER,
            pat_missed INTEGER,
            pat_blocked INTEGER,
            pat_pct FLOAT,
            gwfg_att INTEGER,
            gwfg_made INTEGER,
            gwfg_missed INTEGER,
            gwfg_blocked INTEGER,
            fantasy_points FLOAT
        )
    """)


def create_raw_schedules_table(conn: duckdb.DuckDBPyConnection):
    """Create raw_schedules table matching actual 2025 data (46 columns)"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_schedules (
            -- Game identification (9 columns)
            game_id VARCHAR,
            season INTEGER,
            game_type VARCHAR,
            week INTEGER,
            gameday DATE,
            weekday VARCHAR,
            gametime VARCHAR,
            away_team VARCHAR,
            away_score INTEGER,
            home_team VARCHAR,
            home_score INTEGER,
            location VARCHAR,
            result INTEGER,
            total INTEGER,
            overtime INTEGER,

            -- Game details (31 columns)
            old_game_id VARCHAR,
            away_rest INTEGER,
            home_rest INTEGER,
            away_moneyline FLOAT,
            home_moneyline FLOAT,
            spread_line FLOAT,
            away_spread_odds FLOAT,
            home_spread_odds FLOAT,
            total_line FLOAT,
            under_odds FLOAT,
            over_odds FLOAT,
            div_game INTEGER,
            roof VARCHAR,
            surface VARCHAR,
            temp FLOAT,
            wind FLOAT,
            away_qb_id VARCHAR,
            home_qb_id VARCHAR,
            away_qb_name VARCHAR,
            home_qb_name VARCHAR,
            away_coach VARCHAR,
            home_coach VARCHAR,
            referee VARCHAR,
            stadium_id VARCHAR,
            game_stadium VARCHAR,
            neutral_site INTEGER,
            away_score_q1 INTEGER,
            away_score_q2 INTEGER,
            away_score_q3 INTEGER,
            away_score_q4 INTEGER,
            away_score_overtime INTEGER,
            home_score_q1 INTEGER,
            home_score_q2 INTEGER,
            home_score_q3 INTEGER,
            home_score_q4 INTEGER,
            home_score_overtime INTEGER,
            away_timeouts_remaining INTEGER,
            home_timeouts_remaining INTEGER,
            away_qb_epa FLOAT,
            home_qb_epa FLOAT,
            away_total_epa FLOAT,
            home_total_epa FLOAT,
            away_wp FLOAT,
            home_wp FLOAT,
            def_wp FLOAT
        )
    """)


def create_raw_depth_charts_table(conn: duckdb.DuckDBPyConnection):
    """Create raw_depth_charts table matching actual 2025 data (12 columns)"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_depth_charts (
            season INTEGER,
            club_code VARCHAR,
            week INTEGER,
            game_type VARCHAR,
            depth_team VARCHAR,
            last_name VARCHAR,
            first_name VARCHAR,
            football_name VARCHAR,
            formation VARCHAR,
            gsis_id VARCHAR,
            position VARCHAR,
            elias_id VARCHAR
        )
    """)


def create_raw_rosters_weekly_table(conn: duckdb.DuckDBPyConnection):
    """Create raw_rosters_weekly table matching actual 2025 data (36 columns)"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_rosters_weekly (
            season INTEGER,
            team VARCHAR,
            position VARCHAR,
            depth_chart_position VARCHAR,
            jersey_number INTEGER,
            status VARCHAR,
            full_name VARCHAR,
            first_name VARCHAR,
            last_name VARCHAR,
            birth_date DATE,
            height VARCHAR,
            weight INTEGER,
            college VARCHAR,
            high_school VARCHAR,
            how_acquired VARCHAR,
            gsis_id VARCHAR,
            espn_id VARCHAR,
            sportradar_id VARCHAR,
            yahoo_id VARCHAR,
            rotowire_id VARCHAR,
            update_dt DATE,
            pff_id VARCHAR,
            pfr_id VARCHAR,
            fantasy_data_id VARCHAR,
            sleeper_id VARCHAR,
            years_exp INTEGER,
            headshot_url VARCHAR,
            ngs_position VARCHAR,
            week INTEGER,
            game_type VARCHAR,
            status_description_abbr VARCHAR,
            football_name VARCHAR,
            esb_id VARCHAR,
            gsis_it_id VARCHAR,
            smart_id VARCHAR,
            entry_year INTEGER,
            rookie_year INTEGER
        )
    """)


def create_raw_nextgen_passing_table(conn: duckdb.DuckDBPyConnection):
    """Create raw_nextgen_passing table matching actual 2025 data (29 columns)"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_nextgen_passing (
            season INTEGER,
            season_type VARCHAR,
            week INTEGER,
            player_display_name VARCHAR,
            player_position VARCHAR,
            team_abbr VARCHAR,
            avg_time_to_throw FLOAT,
            avg_completed_air_yards FLOAT,
            avg_intended_air_yards FLOAT,
            avg_air_yards_differential FLOAT,
            aggressiveness FLOAT,
            max_completed_air_distance FLOAT,
            avg_air_yards_to_sticks FLOAT,
            attempts INTEGER,
            pass_yards INTEGER,
            pass_touchdowns INTEGER,
            interceptions INTEGER,
            passer_rating FLOAT,
            completion_percentage FLOAT,
            expected_completion_percentage FLOAT,
            completion_percentage_above_expectation FLOAT,
            avg_air_distance FLOAT,
            max_air_distance FLOAT,
            player_gsis_id VARCHAR,
            player_first_name VARCHAR,
            player_last_name VARCHAR,
            player_jersey_number INTEGER,
            player_short_name VARCHAR,
            pass_yards_above_expectation FLOAT
        )
    """)


def create_raw_nextgen_rushing_table(conn: duckdb.DuckDBPyConnection):
    """Create raw_nextgen_rushing table matching actual 2025 data (22 columns)"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_nextgen_rushing (
            season INTEGER,
            season_type VARCHAR,
            week INTEGER,
            player_display_name VARCHAR,
            player_position VARCHAR,
            team_abbr VARCHAR,
            efficiency FLOAT,
            percent_attempts_gte_eight_defenders FLOAT,
            avg_time_to_los FLOAT,
            rush_attempts INTEGER,
            rush_yards INTEGER,
            expected_rush_yards FLOAT,
            rush_yards_over_expected FLOAT,
            avg_rush_yards FLOAT,
            rush_yards_over_expected_per_att FLOAT,
            rush_pct_over_expected FLOAT,
            player_gsis_id VARCHAR,
            player_first_name VARCHAR,
            player_last_name VARCHAR,
            player_jersey_number INTEGER,
            player_short_name VARCHAR,
            rush_touchdowns INTEGER
        )
    """)


def create_raw_nextgen_receiving_table(conn: duckdb.DuckDBPyConnection):
    """Create raw_nextgen_receiving table matching actual 2025 data (23 columns)"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_nextgen_receiving (
            season INTEGER,
            season_type VARCHAR,
            week INTEGER,
            player_display_name VARCHAR,
            player_position VARCHAR,
            team_abbr VARCHAR,
            avg_cushion FLOAT,
            avg_separation FLOAT,
            avg_intended_air_yards FLOAT,
            percent_share_of_intended_air_yards FLOAT,
            receptions INTEGER,
            targets INTEGER,
            catch_percentage FLOAT,
            yards FLOAT,
            rec_touchdowns INTEGER,
            avg_yac FLOAT,
            avg_expected_yac FLOAT,
            avg_yac_above_expectation FLOAT,
            player_gsis_id VARCHAR,
            player_first_name VARCHAR,
            player_last_name VARCHAR,
            player_jersey_number INTEGER,
            player_short_name VARCHAR
        )
    """)


def create_raw_snap_counts_table(conn: duckdb.DuckDBPyConnection):
    """Create raw_snap_counts table matching actual 2025 data (17 columns)"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_snap_counts (
            player VARCHAR,
            pfr_player_id VARCHAR,
            position VARCHAR,
            team VARCHAR,
            opponent VARCHAR,
            season INTEGER,
            week INTEGER,
            offense_snaps INTEGER,
            offense_pct FLOAT,
            defense_snaps INTEGER,
            defense_pct FLOAT,
            st_snaps INTEGER,
            st_pct FLOAT,
            pfr_game_id VARCHAR,
            game_date DATE,
            player_game_count INTEGER,
            pfr_player_name VARCHAR
        )
    """)


def create_raw_pbp_table(conn: duckdb.DuckDBPyConnection):
    """Create simplified play-by-play table for key metrics only (25 columns + JSON)"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_pbp (
            -- Game identification (7 columns)
            play_id VARCHAR,
            game_id VARCHAR,
            old_game_id VARCHAR,
            home_team VARCHAR,
            away_team VARCHAR,
            season_type VARCHAR,
            week INTEGER,
            posteam VARCHAR,
            posteam_type VARCHAR,
            defteam VARCHAR,

            -- Game situation (10 columns)
            side_of_field VARCHAR,
            yardline_100 INTEGER,
            game_date VARCHAR,
            quarter_seconds_remaining INTEGER,
            half_seconds_remaining INTEGER,
            game_seconds_remaining INTEGER,
            game_half VARCHAR,
            quarter_end INTEGER,
            drive INTEGER,
            sp INTEGER,

            -- Play description (5 columns)
            qtr INTEGER,
            down INTEGER,
            goal_to_go INTEGER,
            time VARCHAR,
            ydstogo INTEGER,

            -- Advanced metrics (3 columns + JSON)
            description VARCHAR,
            play_type VARCHAR,
            yards_gained INTEGER,
            shotgun INTEGER,
            no_huddle INTEGER,
            qb_dropback INTEGER,
            qb_kneel INTEGER,
            qb_spike INTEGER,
            qb_scramble INTEGER,
            pass_length VARCHAR,
            pass_location VARCHAR,
            air_yards FLOAT,
            yards_after_catch INTEGER,
            run_location VARCHAR,
            run_gap VARCHAR,
            field_goal_result VARCHAR,
            kick_distance INTEGER,
            extra_point_result VARCHAR,
            two_point_conv_result VARCHAR,
            home_timeouts_remaining INTEGER,
            away_timeouts_remaining INTEGER,
            timeout INTEGER,
            timeout_team VARCHAR,
            td_team VARCHAR,
            td_player_name VARCHAR,
            td_player_id VARCHAR,
            posteam_timeouts_remaining INTEGER,
            defteam_timeouts_remaining INTEGER,
            total_home_score INTEGER,
            total_away_score INTEGER,
            posteam_score INTEGER,
            defteam_score INTEGER,
            score_differential INTEGER,
            posteam_score_post INTEGER,
            defteam_score_post INTEGER,
            score_differential_post INTEGER,
            no_score_prob FLOAT,
            opp_fg_prob FLOAT,
            opp_safety_prob FLOAT,
            opp_td_prob FLOAT,
            fg_prob FLOAT,
            safety_prob FLOAT,
            td_prob FLOAT,
            extra_point_prob FLOAT,
            two_point_conversion_prob FLOAT,
            ep FLOAT,
            epa FLOAT,
            total_home_epa FLOAT,
            total_away_epa FLOAT,
            total_home_rush_epa FLOAT,
            total_away_rush_epa FLOAT,
            total_home_pass_epa FLOAT,
            total_away_pass_epa FLOAT,
            air_epa FLOAT,
            yac_epa FLOAT,
            comp_air_epa FLOAT,
            comp_yac_epa FLOAT,
            total_home_comp_air_epa FLOAT,
            total_away_comp_air_epa FLOAT,
            total_home_comp_yac_epa FLOAT,
            total_away_comp_yac_epa FLOAT,
            total_home_raw_air_epa FLOAT,
            total_away_raw_air_epa FLOAT,
            total_home_raw_yac_epa FLOAT,
            total_away_raw_yac_epa FLOAT,
            wp FLOAT,
            def_wp FLOAT,
            home_wp FLOAT,
            away_wp FLOAT,
            wpa FLOAT,
            vegas_wpa FLOAT,
            vegas_home_wpa FLOAT,
            home_wp_post FLOAT,
            away_wp_post FLOAT,
            vegas_wp FLOAT,
            vegas_home_wp FLOAT,
            total_home_rush_wpa FLOAT,
            total_away_rush_wpa FLOAT,
            total_home_pass_wpa FLOAT,
            total_away_pass_wpa FLOAT,
            air_wpa FLOAT,
            yac_wpa FLOAT,
            comp_air_wpa FLOAT,
            comp_yac_wpa FLOAT,
            total_home_comp_air_wpa FLOAT,
            total_away_comp_air_wpa FLOAT,
            total_home_comp_yac_wpa FLOAT,
            total_away_comp_yac_wpa FLOAT,
            total_home_raw_air_wpa FLOAT,
            total_away_raw_air_wpa FLOAT,
            total_home_raw_yac_wpa FLOAT,
            total_away_raw_yac_wpa FLOAT,
            punt_blocked INTEGER,
            first_down_rush INTEGER,
            first_down_pass INTEGER,
            first_down_penalty INTEGER,
            third_down_converted INTEGER,
            third_down_failed INTEGER,
            fourth_down_converted INTEGER,
            fourth_down_failed INTEGER,
            incomplete_pass INTEGER,
            touchback INTEGER,
            interception INTEGER,
            punt_inside_twenty INTEGER,
            punt_in_endzone INTEGER,
            punt_out_of_bounds INTEGER,
            punt_downed INTEGER,
            punt_fair_catch INTEGER,
            kickoff_inside_twenty INTEGER,
            kickoff_in_endzone INTEGER,
            kickoff_out_of_bounds INTEGER,
            kickoff_downed INTEGER,
            kickoff_fair_catch INTEGER,
            fumble_forced INTEGER,
            fumble_not_forced INTEGER,
            fumble_out_of_bounds INTEGER,
            solo_tackle INTEGER,
            safety INTEGER,
            penalty INTEGER,
            tackled_for_loss INTEGER,
            fumble_lost INTEGER,
            own_kickoff_recovery INTEGER,
            own_kickoff_recovery_td INTEGER,
            qb_hit INTEGER,
            rush_attempt INTEGER,
            pass_attempt INTEGER,
            sack INTEGER,
            touchdown INTEGER,
            pass_touchdown INTEGER,
            rush_touchdown INTEGER,
            return_touchdown INTEGER,
            extra_point_attempt INTEGER,
            two_point_attempt INTEGER,
            field_goal_attempt INTEGER,
            kickoff_attempt INTEGER,
            punt_attempt INTEGER,
            fumble INTEGER,
            complete_pass INTEGER,
            assist_tackle INTEGER,
            lateral_reception INTEGER,
            lateral_rush INTEGER,
            lateral_return INTEGER,
            lateral_recovery INTEGER,
            passer_player_id VARCHAR,
            passer_player_name VARCHAR,
            passing_yards INTEGER,
            receiver_player_id VARCHAR,
            receiver_player_name VARCHAR,
            receiving_yards INTEGER,
            rusher_player_id VARCHAR,
            rusher_player_name VARCHAR,
            rushing_yards INTEGER,
            lateral_receiver_player_id VARCHAR,
            lateral_receiver_player_name VARCHAR,
            lateral_receiving_yards INTEGER,
            lateral_rusher_player_id VARCHAR,
            lateral_rusher_player_name VARCHAR,
            lateral_rushing_yards INTEGER,
            lateral_sack_player_id VARCHAR,
            lateral_sack_player_name VARCHAR,
            interception_player_id VARCHAR,
            interception_player_name VARCHAR,
            lateral_interception_player_id VARCHAR,
            lateral_interception_player_name VARCHAR,
            punt_returner_player_id VARCHAR,
            punt_returner_player_name VARCHAR,
            lateral_punt_returner_player_id VARCHAR,
            lateral_punt_returner_player_name VARCHAR,
            kickoff_returner_player_name VARCHAR,
            kickoff_returner_player_id VARCHAR,
            lateral_kickoff_returner_player_id VARCHAR,
            lateral_kickoff_returner_player_name VARCHAR,
            punter_player_id VARCHAR,
            punter_player_name VARCHAR,
            kicker_player_name VARCHAR,
            kicker_player_id VARCHAR,
            own_kickoff_recovery_player_id VARCHAR,
            own_kickoff_recovery_player_name VARCHAR,
            blocked_player_id VARCHAR,
            blocked_player_name VARCHAR,
            tackle_for_loss_1_player_id VARCHAR,
            tackle_for_loss_1_player_name VARCHAR,
            tackle_for_loss_2_player_id VARCHAR,
            tackle_for_loss_2_player_name VARCHAR,
            qb_hit_1_player_id VARCHAR,
            qb_hit_1_player_name VARCHAR,
            qb_hit_2_player_id VARCHAR,
            qb_hit_2_player_name VARCHAR,
            forced_fumble_player_1_team VARCHAR,
            forced_fumble_player_1_player_id VARCHAR,
            forced_fumble_player_1_player_name VARCHAR,
            forced_fumble_player_2_team VARCHAR,
            forced_fumble_player_2_player_id VARCHAR,
            forced_fumble_player_2_player_name VARCHAR,
            solo_tackle_1_team VARCHAR,
            solo_tackle_2_team VARCHAR,
            solo_tackle_1_player_id VARCHAR,
            solo_tackle_2_player_id VARCHAR,
            solo_tackle_1_player_name VARCHAR,
            solo_tackle_2_player_name VARCHAR,
            assist_tackle_1_player_id VARCHAR,
            assist_tackle_1_player_name VARCHAR,
            assist_tackle_1_team VARCHAR,
            assist_tackle_2_player_id VARCHAR,
            assist_tackle_2_player_name VARCHAR,
            assist_tackle_2_team VARCHAR,
            assist_tackle_3_player_id VARCHAR,
            assist_tackle_3_player_name VARCHAR,
            assist_tackle_3_team VARCHAR,
            assist_tackle_4_player_id VARCHAR,
            assist_tackle_4_player_name VARCHAR,
            assist_tackle_4_team VARCHAR,
            tackle_with_assist INTEGER,
            tackle_with_assist_1_player_id VARCHAR,
            tackle_with_assist_1_player_name VARCHAR,
            tackle_with_assist_1_team VARCHAR,
            tackle_with_assist_2_player_id VARCHAR,
            tackle_with_assist_2_player_name VARCHAR,
            tackle_with_assist_2_team VARCHAR,
            pass_defense_1_player_id VARCHAR,
            pass_defense_1_player_name VARCHAR,
            pass_defense_2_player_id VARCHAR,
            pass_defense_2_player_name VARCHAR,
            fumbled_1_team VARCHAR,
            fumbled_1_player_id VARCHAR,
            fumbled_1_player_name VARCHAR,
            fumbled_2_player_id VARCHAR,
            fumbled_2_player_name VARCHAR,
            fumbled_2_team VARCHAR,
            fumble_recovery_1_team VARCHAR,
            fumble_recovery_1_player_id VARCHAR,
            fumble_recovery_1_player_name VARCHAR,
            fumble_recovery_1_yards INTEGER,
            fumble_recovery_2_team VARCHAR,
            fumble_recovery_2_player_id VARCHAR,
            fumble_recovery_2_player_name VARCHAR,
            fumble_recovery_2_yards INTEGER,
            sack_player_id VARCHAR,
            sack_player_name VARCHAR,
            half_sack_1_player_id VARCHAR,
            half_sack_1_player_name VARCHAR,
            half_sack_2_player_id VARCHAR,
            half_sack_2_player_name VARCHAR,
            return_team VARCHAR,
            return_yards INTEGER,
            penalty_team VARCHAR,
            penalty_player_id VARCHAR,
            penalty_player_name VARCHAR,
            penalty_yards INTEGER,
            replay_or_challenge INTEGER,
            replay_or_challenge_result VARCHAR,
            penalty_type VARCHAR,
            defensive_two_point_attempt INTEGER,
            defensive_two_point_conv INTEGER,
            defensive_extra_point_attempt INTEGER,
            defensive_extra_point_conv INTEGER,
            safety_player_name VARCHAR,
            safety_player_id VARCHAR,
            season INTEGER,
            cp FLOAT,
            cpoe FLOAT,
            series INTEGER,
            series_success INTEGER,
            series_result VARCHAR,
            order_sequence INTEGER,
            start_time VARCHAR,
            time_of_day VARCHAR,
            stadium VARCHAR,
            weather VARCHAR,
            nfl_api_id VARCHAR,
            play_clock VARCHAR,
            play_deleted INTEGER,
            play_type_nfl VARCHAR,
            special_teams_play INTEGER,
            st_play_type VARCHAR,
            end_clock_time VARCHAR,
            end_yard_line VARCHAR,
            fixed_drive INTEGER,
            fixed_drive_result VARCHAR,
            drive_real_start_time VARCHAR,
            drive_play_count INTEGER,
            drive_time_of_possession VARCHAR,
            drive_first_downs INTEGER,
            drive_inside20 INTEGER,
            drive_ended_with_score INTEGER,
            drive_quarter_start INTEGER,
            drive_quarter_end INTEGER,
            drive_yards_penalized INTEGER,
            drive_start_transition VARCHAR,
            drive_end_transition VARCHAR,
            drive_game_clock_start VARCHAR,
            drive_game_clock_end VARCHAR,
            drive_start_yard_line VARCHAR,
            drive_end_yard_line VARCHAR,
            drive_play_id_started VARCHAR,
            drive_play_id_ended VARCHAR,
            away_score INTEGER,
            home_score INTEGER,
            location VARCHAR,
            result INTEGER,
            total INTEGER,
            spread_line FLOAT,
            total_line FLOAT,
            div_game INTEGER,
            roof VARCHAR,
            surface VARCHAR,
            temp FLOAT,
            wind FLOAT,
            home_coach VARCHAR,
            away_coach VARCHAR,
            stadium_id VARCHAR,
            game_stadium VARCHAR,
            aborted_play INTEGER,
            success INTEGER,
            passer VARCHAR,
            passer_jersey_number INTEGER,
            rusher VARCHAR,
            rusher_jersey_number INTEGER,
            receiver VARCHAR,
            receiver_jersey_number INTEGER,
            pass INTEGER,
            rush INTEGER,
            first_down INTEGER,
            special INTEGER,
            play INTEGER,
            passer_id VARCHAR,
            rusher_id VARCHAR,
            receiver_id VARCHAR,
            name VARCHAR,
            jersey_number INTEGER,
            id VARCHAR,
            fantasy_player_name VARCHAR,
            fantasy_player_id VARCHAR,
            fantasy_pos VARCHAR,
            out_of_bounds INTEGER,
            home_opening_kickoff INTEGER,
            qb_epa FLOAT,
            xyac_epa FLOAT,
            xyac_mean_yardage FLOAT,
            xyac_median_yardage FLOAT,
            xyac_success FLOAT,
            xyac_fd FLOAT,
            xpass FLOAT,
            pass_oe FLOAT
        )
    """)


def create_raw_players_table(conn: duckdb.DuckDBPyConnection):
    """Create player metadata table"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_players (
            player_id VARCHAR PRIMARY KEY,
            player_name VARCHAR,
            position VARCHAR,
            position_group VARCHAR,

            -- Physical Stats
            height VARCHAR,
            weight INTEGER,

            -- Career Info
            years_exp INTEGER,
            team VARCHAR(3),

            -- Draft Information
            draft_year INTEGER,
            draft_round INTEGER,
            draft_pick INTEGER,
            draft_ovr INTEGER,

            -- College
            college VARCHAR,

            -- IDs for joining
            gsis_id VARCHAR,
            espn_id VARCHAR,
            yahoo_id VARCHAR,
            rotowire_id VARCHAR,
            pff_id VARCHAR,
            pfr_id VARCHAR,
            fantasy_data_id VARCHAR,
            sleeper_id VARCHAR,
            esb_id VARCHAR,
            smart_id VARCHAR,

            -- Status
            status VARCHAR
        )
    """)


def create_raw_ftn_charting_table(conn: duckdb.DuckDBPyConnection):
    """Create FTN Charting Data table"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_ftn_charting (
            nflverse_game_id VARCHAR,
            game_id VARCHAR,
            season INTEGER,
            week INTEGER,

            -- Play Details
            play_id VARCHAR,
            drive_id VARCHAR,
            play_type VARCHAR,

            -- Advanced Charting
            n_defense_box INTEGER,
            coverage_type VARCHAR,
            is_motion BOOLEAN,
            is_play_action BOOLEAN,
            is_zone_coverage BOOLEAN
        )
    """)


def create_raw_participation_table(conn: duckdb.DuckDBPyConnection):
    """Create participation data table"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_participation (
            nflverse_game_id VARCHAR,
            play_id VARCHAR,
            gsis_player_id VARCHAR,
            player_name VARCHAR,
            position_group VARCHAR,

            -- Participation flags
            offense BOOLEAN,
            defense BOOLEAN,
            special_teams BOOLEAN
        )
    """)


def create_raw_draft_picks_table(conn: duckdb.DuckDBPyConnection):
    """Create draft picks table"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_draft_picks (
            season INTEGER,
            round INTEGER,
            pick INTEGER,
            team VARCHAR(3),
            pfr_player_name VARCHAR,
            pfr_player_id VARCHAR,
            position VARCHAR,
            age FLOAT,
            college VARCHAR,
            years_as_primary_starter INTEGER,
            career_approximate_value INTEGER,
            draft_approximate_value INTEGER
        )
    """)


def create_raw_combine_table(conn: duckdb.DuckDBPyConnection):
    """Create combine data table"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_combine (
            season INTEGER,
            pfr_id VARCHAR,
            pfr_name VARCHAR,
            pos VARCHAR,
            school VARCHAR,
            height FLOAT,
            weight INTEGER,

            -- Combine Metrics
            forty_yard_dash FLOAT,
            vertical_jump FLOAT,
            bench_press INTEGER,
            broad_jump FLOAT,
            cone_drill FLOAT,
            shuttle_drill FLOAT
        )
    """)


def create_all_raw_tables(conn: duckdb.DuckDBPyConnection):
    """Create all raw data tables using updated 2025 schemas"""

    print("📊 Creating raw data tables...")

    create_raw_player_stats_table(conn)
    print("✅ raw_player_stats table created")

    create_raw_team_stats_table(conn)
    print("✅ raw_team_stats table created")

    create_raw_schedules_table(conn)
    print("✅ raw_schedules table created")

    create_raw_depth_charts_table(conn)
    print("✅ raw_depth_charts table created")

    create_raw_rosters_weekly_table(conn)
    print("✅ raw_rosters_weekly table created")

    create_raw_nextgen_passing_table(conn)
    print("✅ raw_nextgen_passing table created")

    create_raw_nextgen_rushing_table(conn)
    print("✅ raw_nextgen_rushing table created")

    create_raw_nextgen_receiving_table(conn)
    print("✅ raw_nextgen_receiving table created")

    create_raw_snap_counts_table(conn)
    print("✅ raw_snap_counts table created")

    create_raw_pbp_table(conn)
    print("✅ raw_pbp table created")

    create_raw_players_table(conn)
    print("✅ raw_players table created")

    create_raw_ftn_charting_table(conn)
    print("✅ raw_ftn_charting table created")

    create_raw_participation_table(conn)
    print("✅ raw_participation table created")

    create_raw_draft_picks_table(conn)
    print("✅ raw_draft_picks table created")

    create_raw_combine_table(conn)
    print("✅ raw_combine table created")


def create_indexes_for_raw_tables(conn: duckdb.DuckDBPyConnection):
    """Create performance indexes for all raw tables"""

    indexes = [
        # Core table indexes
        "CREATE INDEX IF NOT EXISTS idx_player_stats_player_season_week ON raw_player_stats(player_id, season, week)",
        "CREATE INDEX IF NOT EXISTS idx_player_stats_team_season_week ON raw_player_stats(team, season, week)",
        "CREATE INDEX IF NOT EXISTS idx_player_stats_position ON raw_player_stats(position)",
        "CREATE INDEX IF NOT EXISTS idx_team_stats_team_season_week ON raw_team_stats(team, season, week)",
        "CREATE INDEX IF NOT EXISTS idx_depth_charts_team_season_week ON raw_depth_charts(club_code, season, week)",
        "CREATE INDEX IF NOT EXISTS idx_rosters_weekly_team_season_week ON raw_rosters_weekly(team, season, week)",
        "CREATE INDEX IF NOT EXISTS idx_schedules_season_week ON raw_schedules(season, week)",
        "CREATE INDEX IF NOT EXISTS idx_nextgen_passing_player_season_week ON raw_nextgen_passing(player_gsis_id, season, week)",

        # Next Gen Stats indexes
        "CREATE INDEX IF NOT EXISTS idx_nextgen_rushing_player_season_week ON raw_nextgen_rushing(player_gsis_id, season, week)",
        "CREATE INDEX IF NOT EXISTS idx_nextgen_receiving_player_season_week ON raw_nextgen_receiving(player_gsis_id, season, week)",

        # Snap counts indexes
        "CREATE INDEX IF NOT EXISTS idx_snap_counts_player_season ON raw_snap_counts(pfr_player_id, season, week)",
        "CREATE INDEX IF NOT EXISTS idx_snap_counts_team_season ON raw_snap_counts(team, season, week)",

        # PBP indexes
        "CREATE INDEX IF NOT EXISTS idx_pbp_game_play ON raw_pbp(game_id, play_id)",
        "CREATE INDEX IF NOT EXISTS idx_pbp_season_week ON raw_pbp(season, week)",
        "CREATE INDEX IF NOT EXISTS idx_pbp_passer ON raw_pbp(passer_player_id, season)",

        # Players indexes
        "CREATE INDEX IF NOT EXISTS idx_players_team ON raw_players(team)",
        "CREATE INDEX IF NOT EXISTS idx_players_position ON raw_players(position)",
        "CREATE INDEX IF NOT EXISTS idx_players_draft ON raw_players(draft_year, draft_round)",

        # Advanced tables indexes
        "CREATE INDEX IF NOT EXISTS idx_ftn_game_play ON raw_ftn_charting(game_id, play_id)",
        "CREATE INDEX IF NOT EXISTS idx_participation_player ON raw_participation(gsis_player_id, nflverse_game_id)",
        "CREATE INDEX IF NOT EXISTS idx_draft_season_round ON raw_draft_picks(season, round, pick)",
        "CREATE INDEX IF NOT EXISTS idx_combine_season_pos ON raw_combine(season, pos)"
    ]

    for idx_sql in indexes:
        conn.execute(idx_sql)

    print(f"✅ Created {len(indexes)} performance indexes")


if __name__ == "__main__":
    # Test function to create all schemas
    import duckdb

    print("🧪 Testing consolidated schema creation...")
    conn = duckdb.connect(":memory:")

    try:
        create_all_raw_tables(conn)
        create_indexes_for_raw_tables(conn)
        print("✅ All schemas created successfully!")

        # Show table count
        tables = conn.execute("SHOW TABLES").fetchall()
        print(f"📊 Created {len(tables)} tables total")

    finally:
        conn.close()