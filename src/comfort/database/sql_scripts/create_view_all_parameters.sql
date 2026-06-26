/* Create a view that contains all parameters. Table names are writen down manually. */
CREATE VIEW IF NOT EXISTS v_all AS
	SELECT * FROM 
		(SELECT "barium" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_barium
		UNION ALL
		SELECT "temperature" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_temperature
		UNION ALL
		SELECT "salinity" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_salinity
		UNION ALL
		SELECT "chlorophyll" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_chlorophyll
		UNION ALL
		SELECT "oxygen" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_oxygen
		UNION ALL
		SELECT "turbidity" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_turbidity
		UNION ALL
		SELECT "transmission" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_transmission
		UNION ALL
		SELECT "toc" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_toc
		UNION ALL
		SELECT "tdn" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_tdn
		UNION ALL
		SELECT "tco2" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_tco2
		UNION ALL
		SELECT "silicate" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_silicate
		UNION ALL
		SELECT "sf6" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_sf6
		UNION ALL
		SELECT "psf6" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_psf6
		UNION ALL
		SELECT "phtsinsitutp" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_phtsinsitutp
		UNION ALL
		SELECT "phts27p0" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_phts25p0
		UNION ALL
		SELECT "phosphate" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_phosphate
		UNION ALL
		SELECT "ph" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_ph
		UNION ALL
		SELECT "pco2" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_pco2
		UNION ALL
		SELECT "pcfc12" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_pcfc12
		UNION ALL
		SELECT "pcfc113" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_pcfc113
		UNION ALL
		SELECT "pcfc11" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_pcfc11
		UNION ALL
		SELECT "pccl4" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_pccl4
		UNION ALL
		SELECT "par" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_par
		UNION ALL
		SELECT "o18" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_o18
		UNION ALL
		SELECT "nitrite" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_nitrite
		UNION ALL
		SELECT "nitrate" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_nitrate
		UNION ALL
		SELECT "nitratenitrite" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_nitratenitrite
		UNION ALL
		SELECT "neon" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_neon /*exclude column VALERR*/
		UNION ALL
		SELECT "he3" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_he3 /*exclude column VALERR*/
		UNION ALL
		SELECT "he" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_he /*exclude column VALERR*/
		UNION ALL
		SELECT "h3" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_h3 /*exclude column VALERR*/
		UNION ALL
		SELECT "fluorescence" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_fluorescence
		UNION ALL
		SELECT "don" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_don
		UNION ALL
		SELECT "doc" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_doc
		UNION ALL
		SELECT "din" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_din
		UNION ALL
		SELECT "dic" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_dic
		UNION ALL
		SELECT "cfc12" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_cfc12
		UNION ALL
		SELECT "cfc113" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_cfc113
		UNION ALL
		SELECT "cfc11" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_cfc11
		UNION ALL
		SELECT "cdom" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_cdom
		UNION ALL
		SELECT "ccl4" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_ccl4
		UNION ALL
		SELECT "c14" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_c14  /*exclude column VALERR*/
		UNION ALL
		SELECT "c13" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_c13
		UNION ALL
		SELECT "argon" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_argon
		UNION ALL
		SELECT "aou" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_aou
		UNION ALL
		SELECT "ammonium" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_ammonium
		UNION ALL
		SELECT "alkalinity" as param_name, id, lev_dbar, lev_m, val, pqf1, pqf2, sqf, bottle_number, profile_number, profile_best, units_id, instrument_id, latitude, longitude, dateandtime FROM v_alkalinity)
	;
