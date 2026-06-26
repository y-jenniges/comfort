/* Create one view for each parameter that additionally contains latitude, longitude, dateandtime information.
Table names are writen down manually. */

create view IF NOT EXISTS v_alkalinity as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_alkalinity as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_ammonium as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_ammonium as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_aou as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_aou as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_argon as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_argon t join station as s on t.id=s.id;

create view IF NOT EXISTS v_barium as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_barium as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_c13 as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_c13 as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_c14 as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_c14 as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_ccl4 as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_ccl4 as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_cdom as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_cdom as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_cfc11 as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_cfc11 as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_cfc113 as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_cfc113 as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_cfc12 as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_cfc12 as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_chlorophyll as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_chlorophyll as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_dic as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_dic as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_din as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_din as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_doc as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_doc as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_don as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_don as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_fluorescence as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_fluorescence as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_h3 as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_h3 as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_he as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_he as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_he3 as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_he3 as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_neon as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_neon as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_nitrate as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_nitrate as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_nitratenitrite as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_nitratenitrite as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_nitrite as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_nitrite as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_o18 as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_o18 as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_oxygen as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_oxygen as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_par as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_par as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_pccl4 as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_pccl4 as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_pcfc11 as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_pcfc11 as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_pcfc113 as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_pcfc113 as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_pcfc12 as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_pcfc12 as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_pco2 as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_pco2 as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_ph as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_ph as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_phosphate as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_phosphate as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_phts25p0 as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_phts25p0 as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_phtsinsitutp as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_phtsinsitutp as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_psf6 as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_psf6 as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_sf6 as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_sf6 as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_salinity as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_salinity as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_silicate as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_silicate as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_tco2 as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_tco2 as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_tdn as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_tdn as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_temperature as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_temperature as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_toc as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_toc as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_transmission as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_transmission as t join station as s on t.id=s.id;

create view IF NOT EXISTS v_turbidity as
    select t.*, s.latitude, s.longitude, s.dateandtime
    from p_turbidity as t join station as s on t.id=s.id;
