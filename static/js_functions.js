function get_val_from_id(id){
    for (const e of data) {
      // ...use `element`...
        if (e.cbsid==id){
            return e.Value;
        };
    };
};

$.fn.owners_chart = function(x_var, y_var){
    let x_data = [];
    let y_data = [];
    $.each(owners, function(i, v){
        x_data.push(v[x_var]);
        y_data.push(v[y_var]);
    })
    owners_data = [
        {
            type: 'bar',
            x:x_data,
            y:y_data,
            
        }
    ]
    layout = {title: "Title", height: 400, width: 1000, margin: {t:30}},
    Plotly.newPlot("owners_chart", owners_data, layout, {displayModeBar: false})    
}
$.fn.z_players = function(){
    let x_data = [];
    let y_data = [];
    let hover_data = [];
    let p_id = [];
    let color_map = [];
    var j = 0;
    $.each(data, function(i, v){
        if (j<440){
            x_data.push(j);
            y_data.push(data[i]['curValue']);
            hover_data.push(data[i]['Name']+'<br>ID: '+data[i]['cbsid']+'<br>Value: $'+data[i]['Value']+'<br>Market: $'+data[i]['curValue']);
            if (data[i]['Owner']){
                color_map.push('gray');
            } else {
                if (data[i]['surplus']>6){
                    color_map.push('green');
                } else {
                    color_map.push('lightblue');
                }
            }
        }
        j += 1
    })

    z_scatter_data = [
        {
            type: 'scatter',
            x:x_data,
            y:y_data,
            text:hover_data,
            mode:'markers',
            customtext:p_id,
            marker: { color: color_map, size:8, opacity:.6, line:{color:'black', width:1} },
            hovertemplate: "%{text}"
            
        }
    ]
    layout = {title: "Z List", height: 500, width: 1350, margin: {l:15, t:30}},
    Plotly.newPlot("z_players_chart", z_scatter_data, layout, {displayModeBar: false})
}

$.fn.tiers = function(){
    let x_data = [];
    let y_data = [];
    let hover_data = [];
    let p_id = [];
    let color_map = [];
    var j = 0;
    var sortOrder = ['C', '1B', '2B', '3B', 'SS', 'OF', 'DH', 'SP', 'RP'] 
    // Sort the data by Primary_Pos then curValue
    var sortedData = data.sort(function(a, b) {
        var posA = sortOrder.indexOf(a.Primary_Pos);
        var posB = sortOrder.indexOf(b.Primary_Pos);

        if (posA === posB) {
            // Secondary sort by curValue
            return b.curValue - a.curValue;
        }

        return posA - posB;
    });
    $.each(sortedData, function(i, v){
        if (v.z>-2.5){
            x_data.push(v.Primary_Pos);
            y_data.push(sortedData[i]['curValue']);
            hover_data.push(sortedData[i]['Name']+'<br>ID: '+sortedData[i]['cbsid']+'<br>Value: $'+sortedData[i]['Value']+'<br>Market Value: $'+sortedData[i]['curValue']);
            if (sortedData[i]['Owner']){
                color_map.push('gray');
            } else {
                if (sortedData[i]['surplus']>6){
                    color_map.push('green');
                } else {
                    color_map.push('lightblue');
                }
            }
        }
        //j += 1
    })

    tiers_data = [
        {
            type: 'scatter',
            x: x_data,
            y: y_data,
            text: hover_data,
            //hovermode:'closest',
            mode:'markers',
            customtext: p_id,
            marker: { color: color_map, opacity:.5, size:9, line:{color:'gray', width:1}},
            hovertemplate: "%{text}"
            
        }
    ]
    layout = {title: "Positional Tiers", hovermode:'closest', height: 500, width: 1300, margin: {l:20, t:30}, yaxis: {range: [-15,45]},
    shapes: [
            {type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 35.7, y1: 35.7, yref: 'y', line: {color: 'red', width: 1, dash: 'dash'}},
            {type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 27.6, y1: 27.6, yref: 'y', line: {color: 'red', width: 1, dash: 'dash'}},
            {type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 24.54, y1: 24.54, yref: 'y', line: {color: 'red', width: 1, dash: 'dash'}},
            {type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 19.75, y1: 19.75, yref: 'y', line: {color: 'red', width: 1, dash: 'dash'}},
            {type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 16.7, y1: 16.7, yref: 'y', line: {color: 'red', width: 1, dash: 'dash'}},
            //{type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 15.03, y1: 15.03, yref: 'y', line: {color: 'red', width: 1, dash: 'dash'}},
            //{type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 13.5, y1: 13.5, yref: 'y', line: {color: 'red', width: 1, dash: 'dash'}},
            //{type: 'line', x0: -0.25, x1: 0.25, y0: 12.28, y1: 12.28, yref: 'y', line: {color: 'red', width: 1, dash: 'dash'}},
        ],
    },
    Plotly.newPlot("tiers_chart", tiers_data, layout, {displayModeBar: false})
}

$.fn.paid_histogram = function(){
    let x_data = [];
    let y_data = [];
    let hover_data = [];
    let p_id = [];
    let color_map = [];
    var j = 0;
    $.each(paid_hist_data, function(i, v){
        y_data.push(v)
    })
    var trace1 = 
        {
            name:'Historical',
            type: 'bar',
            x:['1-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40+'],
            y:[5.27, 7.4, 4.07, 2.13, 1.87, 1.07, .6, .27, .33],
            opacity:.75,
            marker:{color:'gray'},
        }
    
    var trace2 = 
        {
            name:'Lima Time',
            type: 'bar',
            x:['1-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40+'],
            y:y_data,
            opacity:.75,
            marker:{color:'blue'},
        }
    
    pd_hist_data = [trace1, trace2];
    layout = {title: "Historical Paid by Amount", hovermode:'closest', height: 400, width: 1050, margin: {l:20, t:30}},
    Plotly.newPlot("paid_hist_chart", pd_hist_data, layout, {displayModeBar: false})
}


$.fn.update_player_stats_window = function(selected_index){
    let tbl_html = '<table class="table draft-table" id="player_table">'
    +'<thead><tr><th>cbsid</th><th>Name</th><th>Team</th><th>Pos</th><th>Age</th><th>Proj Value</th><th>Market</th><th>CBS</th><th>FG</th><th>Val_ly</th><th>Z</th><th>Vol</th><th>Skew</th></tr></thead>'
    +'<tr><td>'+data[selected_index]['cbsid']+'</td>'
    +'<td>'+data[selected_index]['Name']+'</td>'
    +'<td>'+data[selected_index]['Team']+'</td>'
    +'<td>'+data[selected_index]['Pos']+'</td>'
    +'<td>'+data[selected_index]['player_age_ly']+'</td>'
    +'<td class="text-value">$'+data[selected_index]['Value']+'</td>'
    +'<td>'+data[selected_index]['curValue']+'</td>'
    +'<td>'+data[selected_index]['CBS']+'</td>'
    +'<td>'+data[selected_index]['Dollars']+'</td>'
    +'<td>'+data[selected_index]['Value_ly']+'</td>'
    +'<td>'+data[selected_index]['z']+'</td>'
    +'<td>'+ (data[selected_index]['Vol'] != null ? data[selected_index]['Vol'].toFixed(2) : 'N/A') +'</td>'
    +'<td>'+ (data[selected_index]['Skew'] != null ? data[selected_index]['Skew'].toFixed(2) : 'N/A') +'</td>'
    +'</tr></table>'
    if ((data[selected_index]['Primary_Pos']=='SP') || (data[selected_index]['Primary_Pos']=='RP')){
        tbl_html += `<table class="table draft-table table-striped"><thead><tr><th>Type</th><th>IP</th><th>ERA</th><th>WHIP</th><th>K</th><th>QS</th><th>S+H</th>
        <th>K-BB%</th><th>K/9</th><th>Velo</th><th>IVB</th><th>woba_diff</th><th>Whiff%</th><th>EV</th></tr></thead>`
        +'<tr><td>Proj</td>'
        +'<td>'+data[selected_index]['IP']+'</td>'
        +'<td>'+data[selected_index]['ERA']+'</td>'
        +'<td>'+data[selected_index]['WHIP']+'</td>'
        +'<td>'+data[selected_index]['SO']+'</td>'
        +'<td>'+data[selected_index]['QS']+'</td>'
        +'<td>'+data[selected_index]['SvHld']+'</td>'
        +'</tr><tr><td>'+previousYear+'</td>'
        +'<td>'+(data[selected_index]['p_out_ly']/3).toFixed(1)+'</td>'
        +'<td>'+data[selected_index]['p_era_ly']+'</td>'
        +'<td>'+Number(data[selected_index]['p_whip_ly']).toFixed(2)+'</td>'
        +'<td>'+data[selected_index]['p_strikeout_ly']+'</td>'
        +'<td>'+data[selected_index]['p_quality_start_ly']+'</td>'
        +'<td>'+data[selected_index]['p_SvHld_ly']+'</td>'
        +'<td>'+(data[selected_index]['K-BB%_ly']*100).toFixed(1)+'%</td>'
        +'<td>'+Number(data[selected_index]['K/9_ly']).toFixed(1)+'</td>'
        +'<td>'+data[selected_index]['fastball_avg_speed_ly']+'</td>'
        +'<td>'+data[selected_index]['fastball_avg_break_z_induced_ly']+'</td>'
        +'<td>'+data[selected_index]['woba_diff_ly']+'</td>'
        +'<td>'+data[selected_index]['whiff_percent_ly']+'%</td>'
        +'<td>'+data[selected_index]['exit_velocity_avg_ly']+'</td>'
        +'</tr>'
        // 2 years ago
        +'<tr><td>'+twoYearsAgo+'</td>'
        +'<td>'+(data[selected_index]['p_out_2ly']/3).toFixed(1)+'</td>'
        +'<td>'+data[selected_index]['p_era_2ly']+'</td>'
        +'<td>'+Number(data[selected_index]['p_whip_2ly']).toFixed(2)+'</td>'
        +'<td>'+data[selected_index]['p_strikeout_2ly']+'</td>'
        +'<td>'+data[selected_index]['p_quality_start_2ly']+'</td>'
        +'<td>'+data[selected_index]['p_SvHld_2ly']+'</td>'
        +'<td>'+(data[selected_index]['K-BB%_2ly']*100).toFixed(1)+'%</td>'
        +'<td>'+Number(data[selected_index]['K/9_2ly']).toFixed(1) +'</td>'
        +'<td>'+data[selected_index]['fastball_avg_speed_2ly']+'</td>'
        +'<td>'+data[selected_index]['fastball_avg_break_z_induced_2ly']+'</td>'
        +'<td>'+data[selected_index]['woba_diff_2ly']+'</td>'
        +'<td>'+data[selected_index]['whiff_percent_2ly']+'%</td>'
        +'<td>'+data[selected_index]['exit_velocity_avg_2ly']+'</td>'
        +'</tr></table>'
    } else {
        tbl_html += `<table class="table draft-table table-striped"><thead><tr><th>Type</th><th>PA</th><th>wOBAdiff</th><th>xBA</th><th>BA</th><th>HR</th><th>SB</th><th>R</th><th>RBI</th>
        <th>Brl%</th><th>EV</th><th>Spd</th><th>Whiff%</th><th>wRC+</th></tr></thead>`
        +'<tr><td>Proj</td>'
        +'<td>'+ (data[selected_index]['PA'] != null ? data[selected_index]['PA'].toFixed(1) : 'N/A')+'</td>'
        +'<td>-</td><td>-</td>'
        +'<td>'+data[selected_index]['BA']+'</td>'
        +'<td>'+data[selected_index]['HR']+'</td>'
        +'<td>'+data[selected_index]['SB']+'</td>'
        +'<td>'+data[selected_index]['R']+'</td>'
        +'<td>'+data[selected_index]['RBI']+'</td>'
        +'</tr><tr><td>'+previousYear+'</td>'
        +'<td>'+data[selected_index]['pa_ly']+'</td>'
        +'<td>'+data[selected_index]['woba_diff_ly']+'</td>'
        +'<td>'+data[selected_index]['xba_ly']+'</td>'
        +'<td>'+data[selected_index]['batting_avg_ly']+'</td>'
        +'<td>'+data[selected_index]['home_run_ly']+'</td>'
        +'<td>'+data[selected_index]['r_total_stolen_base_ly']+'</td>'
        +'<td>'+data[selected_index]['r_run_ly']+'</td>'
        +'<td>'+data[selected_index]['b_rbi_ly']+'</td>'
        +'<td>'+data[selected_index]['barrel_batted_rate_ly']+'</td>'
        +'<td>'+data[selected_index]['exit_velocity_avg_ly']+'</td>'
        +'<td>'+data[selected_index]['sprint_speed_ly']+'</td>'
        +'<td>'+data[selected_index]['whiff_percent_ly']+'%</td>'
        +'<td>'+ (data[selected_index]['wRC+'] != null ? data[selected_index]['wRC+'].toFixed(0) : 'N/A')+'</td>'
        // 2 years ago
        +'</tr><tr><td>'+twoYearsAgo+'</td>'
        +'<td>'+data[selected_index]['pa_2ly']+'</td>'
        +'<td>'+data[selected_index]['woba_diff_2ly']+'</td>'
        +'<td>'+data[selected_index]['xba_2ly']+'</td>'
        +'<td>'+data[selected_index]['batting_avg_2ly']+'</td>'
        +'<td>'+data[selected_index]['home_run_2ly']+'</td>'
        +'<td>'+data[selected_index]['r_total_stolen_base_2ly']+'</td>'
        +'<td>'+data[selected_index]['r_run_2ly']+'</td>'
        +'<td>'+data[selected_index]['b_rbi_2ly']+'</td>'
        +'<td>'+data[selected_index]['barrel_batted_rate_2ly']+'</td>'
        +'<td>'+data[selected_index]['exit_velocity_avg_2ly']+'</td>'
        +'<td>'+data[selected_index]['sprint_speed_2ly']+'</td>'
        +'<td>'+data[selected_index]['whiff_percent_2ly']+'</td>'
        +'<td></td>'
        +'</tr></table>'
    }
    
    $("#player_stats_window").html(tbl_html);
    
    console.log(data[selected_index]['Primary_Pos']);

}

$.fn.create_radar_chart = function(selected){
    $("#projected_stats_table tr").hide();
    $("#projected_stats_table tr:first").show();
    $("#statcast_stats_table tr").hide();
    $("#statcast_stats_table tr:first").show();
    $("#"+selected).show();
    $("#"+selected+'_sc').show();
    $.each(data, function(i, v) {
            if (v.cbsid == selected) {
                selected_index = i
                return;
            }
        });
    $.fn.update_player_stats_window(selected_index);
    $("#radar_chart_player_name").text(data[selected_index]['Name']);
    let position = data[selected_index]['Primary_Pos'];
    if ((position == 'SP') | (position =='RP')){
        radar_data = [{
            type: 'scatterpolar',
            r: [data[selected_index]['zERA'], data[selected_index]['zWHIP'], data[selected_index]['zQS'], data[selected_index]['zSO'], data[selected_index]['zSvHld'], data[selected_index]['zERA']],
            theta: ['ERA','WHIP','QS', 'SO', 'SvHld', 'ERA'],
            fill: 'toself'
            }]
    } else {
        radar_data = [{
            type: 'scatterpolar',
            r: [data[selected_index]['zBA'], data[selected_index]['zHR'], data[selected_index]['zR'], data[selected_index]['zRBI'], data[selected_index]['zSB'], data[selected_index]['zBA']],
            theta: ['BA','HR','R', 'RBI', 'SB', 'BA'],
            fill: 'toself'
            }]
    }
    layout = {
        height: 300,
        polar: {
            radialaxis: {
            visible: true,
            range: [-3, 3]
            },
        margin: { l:0, r:0, t:0, b:0, pad:0}
    },
    showlegend: false
    }
    Plotly.newPlot("radar_chart", radar_data, layout, {displayModeBar: false})
}

function bid_amounts(id){
    const val = get_val_from_id(id);
    for (key in owners){
        let bid = val*owners[key]["Cash"];
        let o = owners[key]['Owner'].replace(' ','_');
        $("#"+o+"_meter").text(bid.toFixed(0));
    }
}

var lastBidResult = null;
const previousYear = new Date().getFullYear() - 1;
const twoYearsAgo = new Date().getFullYear() - 2;

$(document).ready(function(){
    

    if (redirectStatus =='unrosterable'){
        alert('Unable to roster last drafted player')
    }
    var el = $("#team_input")
    t = `<div class="row">`
    c = 1;
    for (let tm of owner_list){
    if (c==3 | c==6 | c==9 | c==12){
        t+= `<div class="col"><input type="radio" name="owner" value="${tm}">
            <label for="team${c}">${tm.slice(0,11)}</label></div></div><div class="row">`
        c+=1;
    } else {
        t+= `<div class="col"><input type="radio" name="owner" value="${tm}">
            <label for="team${c}">${tm.slice(0,11)}</label></div>`
        c += 1;
    }
    }
    t +=  `</div>`
    el.html(t)

    $("#projected_stats_table tr").hide();
    $("#projected_stats_table tr:first").show();
    $("#statcast_stats_table tr").hide();
    $("#statcast_stats_table tr:first").show();
    $.fn.z_players();
    $.fn.owners_chart('Owner', '$ Left');
    $.fn.tiers();
    $.fn.paid_histogram();
    
    $("input[name='cbsid']").on('focusout', function(e){
        var selected = $(this).val();
        $(this).create_radar_chart(selected);
        bid_amounts(selected);
        
        $.get("/draft/sims/"+selected, function(resp, status){
            //alert("Data: " + resp + "\nStatus: " + status);
            resp = JSON.parse(resp);
            let names = [];
            let values = [];
            let html_response = "";
            $.each(resp, function(i, v){
                values.push(v.Name +', '+ v.Value.toFixed(1).toString() + '<br>');
            })
            $("#sims").html(values.join(''));
        });
        
        // Simulate auction bids
        pdata = 0
        $.each(data, function(k,v){
            if (v['cbsid']==selected){
                pdata = k
            }
        })
        jsonData = {'owners':owners, 'player_data':data[pdata], 'roster':roster}

        $.ajax({
            url: '/draft/get_bids',
            type: 'POST',
            data: JSON.stringify(jsonData),
            contentType: 'application/json',
            dataType: 'json',
            success: function(response) {
                lastBidResult = response;
                $("#bidWinner").html(
                    '<strong>' + response.winner + '</strong> $' + response.price
                    + ' <span class="text-muted-custom">(would pay up to $' + response.max_willingness + ')</span>'
                );
            },
            error: function(xhr, status, error) {
                console.error('get_bids failed:', status, error, xhr.responseText);
                $("#bidWinner").html('<span class="text-negative">Auction sim error: ' + error + '</span>');
            }
        });

    });

    $("#bid_form").submit(function(){
        $("#error_msg").hide();
        var player_id = $("input[name='cbsid']").val();
        var bid_winner = $('input[name="owner"]:checked').val();
        var price_val = $("#price_entry").val();
        var supp_round = $("#supp_entry").val();

        if (player_id==""){
            alert("Must select a player");
            $("#error_msg").text('Choose a player').show();
            return false;
        }
        if (!bid_winner){
            alert('Must select an owner');
            $("#error_msg").text('Choose a team').show();
            return false;
        }
        if (price_val<0 | price_val=='') {
            alert('Must select a valid price')
            $("#error_msg").text('Enter a valid price').show();
            return false;
        }
        if (supp_round < 0 | supp_round > 10){
            alert('Must select a supp round 0 to 10')
            $("#error_msg").text('Enter a valid supp').show();
            return false;
        }
        if (price_val >0 & supp_round > 0){
            alert(`Either price or supp must be 0`);
            $("#error_msg").text('Either price or supp must be 0').show();
            return false;
        }
    })
    
    $("#acceptAuctionBid").click(function(){
        if (!lastBidResult) {
            alert('No auction result to accept');
            return;
        }
        var cbsid = $("#player_select").val();
        var team = lastBidResult.winner;
        var price = lastBidResult.price;
        url = `/draft/update_bid?cbsid=${cbsid}&owner=${team}&price=${price}&supp=0`;
        window.location = url;
    })

    $("#button").click(function(){
        var v = $("#player_list").val();
        $.get("/draft/"+v, function(data, status){
            alert("Data: " + data + "\nStatus: " + status);
        });
    });
    $("#button-1").click(function(){
        $.fn.owners_chart('Owner', 'Pts');
        $(this).addClass('active').siblings().removeClass('active');
    });
    $("#button-2").click(function(){
        $.fn.owners_chart('Owner', '$ Left');
        $(this).addClass('active').siblings().removeClass('active');
    });
    $("#button-3").click(function(){
        $.fn.owners_chart('Owner', 'z');
        $(this).addClass('active').siblings().removeClass('active');
    });
    $("#button-4").click(function(){
        $.fn.owners_chart('Owner', '$/unit');
        $(this).addClass('active').siblings().removeClass('active');
    });
    $("#button-5").click(function(){
        $.fn.owners_chart('Owner', 'Drafted');
        $(this).addClass('active').siblings().removeClass('active');
    }); 
    $("#rosterTable").on("click", "td", function() {
        var p_name = $(this).text();
        $.each(data, function(i, v) {
            if (v.Name == p_name) {
                var tr_id = v.cbsid
                $("#player_select").val(tr_id);
                $(this).create_radar_chart(tr_id);
                bid_amounts(tr_id);
                return tr_id;
            }
        });
    });
    $("#drafted_scroll").on("click", "td", function() {
        var p_name = $(this).text();
        $.each(data, function(i, v) {
            if (v.Name == p_name) {
                var tr_id = v.cbsid
                $("#player_select").val(tr_id);
                $(this).create_radar_chart(tr_id);
                bid_amounts(tr_id);
                return tr_id;
            }
        });
    });
    $("#surplus").on("click", "td", function() {
        var p_name = $(this).text();
        $.each(data, function(i, v) {
            if (v.Name == p_name) {
                var tr_id = v.cbsid
                $("#player_select").val(tr_id);
                $(this).create_radar_chart(tr_id);
                bid_amounts(tr_id);
                return tr_id;
            }
        });
    });
    $("#stat_tiers").on("click", "td", function() {
        var p_name = $(this).text();
        $.each(data, function(i, v) {
            if (v.Name == p_name) {
                var tr_id = v.cbsid
                $("#player_select").val(tr_id);
                $(this).create_radar_chart(tr_id);
                bid_amounts(tr_id);
                return tr_id;
            }
        });
    });
    // Lima Time highlighting now handled by .my-team-row CSS class in the template


    document.getElementById("z_players_chart").on('plotly_click', function(data){
        var txt = data.points[0].text.split("<br>")
        console.log(txt);
        $("#player_select").val(txt[1].substring(4));
        $.fn.create_radar_chart(txt[1].substring(4));
        bid_amounts(txt[1].substring(4));
    });
    document.getElementById("tiers_chart").on('plotly_click', function(data){
        var txt = data.points[0].text.split("<br>")
        $("#player_select").val(txt[1].substring(4));
        $.fn.create_radar_chart(txt[1].substring(4));
        bid_amounts(txt[1].substring(4));
    });
})
