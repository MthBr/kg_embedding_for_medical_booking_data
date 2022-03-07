#!/bin/bash

file_sub=$1

echo "########################################"
echo "Substitution file: " $file_sub
echo "########################################"


array=(
    'referall-modified-id,indx_impegnativa'
    'local-health-department-id,sa_asl'
    'patient-age,sa_eta_id'
    'gender,sa_sesso_id'
    'encrypted-nin-id,sa_ass_cf'
    'nuts-istat-code,istat-comune-id'
    'practitioner-id,sa_med_id'
    'booking-agent-id,sa_utente_id'
    'priority-code,sa_classe_priorita'
    'exemption-code,sa_ese_id_lk'
    'referral-id,sa_impegnativa_id'
    'refined-health-service-id,indx_prestazione'
    'health-service-id,sa_pre_id'
    'refined-medical-branch-id,indx_branca'
    'medical-branch-id,sa_branca_id'
    'health-service-description,descrizione-prestazione'
    'official-branch-description,descrizione-ufficiale'
    'branch-description,descrizione-branca'
    'number-of-health-services,sa_num_prestazioni'
    'referral-centre-id,sa_uop_codice_id'
    'booking-type,sa_is_ad'
    'appointment-encoding,sa_dti_id'
    'event-date,data'
    'last-reservation-change-date,sa_data_ins'
    'reservation-date,sa_data_pren'
    'booked-date,sa_data_app'
    'referral-date,sa_data_prescr'
    'ledger-entry-date,sa_data_ins-pagamento'
    'transaction-date,sa_data_mov'
    'referred-patient,assistito-indicato'
    'patient,assistito'
    'practitioner,medico-prescrittore'
    'referrer,prescrivente'
    'booking-staff,operatore-utente'
    'booking-agent,operatore-che-prenota'
    'deleting-agent,operatore-che-elimina'
    'updating-agent,operatore-che-aggiorna'
    'referred-medical-branch,branca-indicata'
    'medical-branch,branca'
    'reserved-health-service,prestazione-prenotata'
    'prescribed-health-service,prestazione-prescritta'
    'health-service-provision,prestazione'
    'appointment-provider,UOP'
    'referring-centre,erogatore-ambulatorio'
    'referral,impegnativa'
    'booked-referall,appuntamento'
    'reservation,prenotazione'
)

for elem in "${array[@]}"
do
    elemIN=(${elem//,/ })

    sed -i "s/${elemIN[0]}\([;, ]\)/${elemIN[1]}\1/g" $file_sub
done

echo "########################################"
echo "Finished!"
echo "########################################"
