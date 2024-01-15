function get_val_from_id(id){
    for (const e of data) {
      // ...use `element`...
        if (e.playerid==id){
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
            y_data.push(data[i]['Value']);
            hover_data.push(data[i]['Name']+'<br>ID: '+data[i]['playerid']+'<br>Value: $'+data[i]['Value']);
            if (data[i]['Owner']){
                color_map.push('gray');
            } else {
                color_map.push('lightblue');
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
            marker: { color: color_map },
            hovertemplate: "%{text}"
            
        }
    ]
    layout = {title: "Z List", height: 400, width: 1350, margin: {l:15, t:30}},
    Plotly.newPlot("z_players_chart", z_scatter_data, layout, {displayModeBar: false})
}

$.fn.tiers = function(){
    let x_data = [];
    let y_data = [];
    let hover_data = [];
    let p_id = [];
    let color_map = [];
    var j = 0;
    $.each(data, function(i, v){
        if (v.z>-1.5){
            x_data.push(v.Primary_Pos);
            y_data.push(data[i]['Value']);
            hover_data.push(data[i]['Name']+'<br>ID: '+data[i]['playerid']+'<br>Market Value: $'+data[i]['Value']);
            if (data[i]['Owner']){
                color_map.push('gray');
            } else {
                color_map.push('lightblue');
            }
        }
        //j += 1
    })

    tiers_data = [
        {
            type: 'scatter',
            x:x_data,
            y:y_data,
            text:hover_data,
            //hovermode:'closest',
            mode:'markers',
            customtext:p_id,
            marker: { color: color_map },
            hovertemplate: "%{text}"
            
        }
    ]
    layout = {title: "Positional Tiers", hovermode:'closest', height: 400, width: 1050, margin: {l:20, t:30}},
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
    let tbl_html = '<table class="table" id="player_table">'
    +'<tr><thead><th>playerid</th><th>Name</th><th>Team</th><th>Pos</th><th>Age</th><th>Proj Value</th><th>Market</th><th>CBS</th><th>FG</th><th>Val_ly</th><th>Z</th></thead></tr>'
    +'<tr><td>'+data[selected_index]['playerid']+'</td>'
    +'<td>'+data[selected_index]['Name']+'</td>'
    +'<td>'+data[selected_index]['Team']+'</td>'
    +'<td>'+data[selected_index]['Pos']+'</td>'
    +'<td>'+data[selected_index]['Age']+'</td>'
    +'<td><font color="red">$'+data[selected_index]['Value']+'</font></td>'
    +'<td>'+data[selected_index]['curValue']+'</td>'
    +'<td>'+data[selected_index]['CBS']+'</td>'
    +'<td>'+data[selected_index]['Dollars']+'</td>'
    +'<td>'+data[selected_index]['Value_ly']+'</td>'
    +'<td>'+data[selected_index]['z']+'</td>'
    +'</tr></table><br>'
    
    if ((data[selected_index]['Primary_Pos']=='SP') || (data[selected_index]['Primary_Pos']=='RP')){
        tbl_html += '<table class="table table-striped"><tr><thead><th>Type</th><th>IP</th><th>ERA</th><th>WHIP</th><th>K</th><th>W</th><th>S+H</th><th>FIP</th><th>HR/9</th><th>K/9</th><th>BB/9</th><th>K%</th><th>BB%</th><th>SIERA</th><th>ERA-</th><th>CSW%</th><th>FBv</th></thead></tr>'
        +'<tr><td>Proj</td>'
        +'<td>'+data[selected_index]['IP']+'</td>'
        +'<td>'+data[selected_index]['ERA']+'</td>'
        +'<td>'+data[selected_index]['WHIP']+'</td>'
        +'<td>'+data[selected_index]['SO']+'</td>'
        +'<td>'+data[selected_index]['W']+'</td>'
        +'<td>'+data[selected_index]['Sv+Hld']+'</td>'
        +'</tr><tr><td>2022</td>'
        +'<td>'+data[selected_index]['IP_ly']+'</td>'
        +'<td>'+data[selected_index]['ERA_ly']+'</td>'
        +'<td>'+data[selected_index]['WHIP_ly']+'</td>'
        +'<td>'+data[selected_index]['SO_ly']+'</td>'
        +'<td>'+data[selected_index]['W_ly']+'</td>'
        +'<td>'+data[selected_index]['Sv+Hld_ly']+'</td>'
        +'<td>'+data[selected_index]['FIP']+'</td>'
        +'<td>'+data[selected_index]['HR/9']+'</td>'
        +'<td>'+data[selected_index]['K/9']+'</td>'
        +'<td>'+data[selected_index]['BB/9']+'</td>'
        +'<td>'+data[selected_index]['K%']+'</td>'
        +'<td>'+data[selected_index]['BB%']+'</td>'
        +'<td>'+data[selected_index]['SIERA']+'</td>'
        +'<td>'+data[selected_index]['ERA-']+'</td>'
        +'<td>'+data[selected_index]['CSW%']+'</td>'
        +'<td>'+data[selected_index]['FBv']+'</td>'
        +'</tr></table>'
    } else {
        tbl_html += '<table class="table table-striped"><tr><thead><th>Type</th><th>PA</th><th>xwOBA</th><th>xBA</th><th>BA</th><th>HR</th><th>SB</th><th>R</th><th>RBI</th><th>Brl%</th><th>BB%</th><th>K%</th><th>Cont%</th><th>O-Swing%</th><th>HardHit%</th><th>maxEV</th><th>EV</th></thead></tr>'
        +'<tr><td>Proj</td>'
        +'<td>'+data[selected_index]['PA']+'</td>'
        +'<td>-</td><td>-</td>'
        +'<td>'+data[selected_index]['BA']+'</td>'
        +'<td>'+data[selected_index]['HR']+'</td>'
        +'<td>'+data[selected_index]['SB']+'</td>'
        +'<td>'+data[selected_index]['R']+'</td>'
        +'<td>'+data[selected_index]['RBI']+'</td>'
        +'</tr><tr><td>2022</td>'
        +'<td>'+data[selected_index]['PA_ly']+'</td>'
        +'<td>'+data[selected_index]['xwOBA']+'</td>'
        +'<td>'+data[selected_index]['xBA']+'</td>'
        +'<td>'+data[selected_index]['BA_ly']+'</td>'
        +'<td>'+data[selected_index]['HR_ly']+'</td>'
        +'<td>'+data[selected_index]['SB_ly']+'</td>'
        +'<td>'+data[selected_index]['R_ly']+'</td>'
        +'<td>'+data[selected_index]['RBI_ly']+'</td>'
        +'<td>'+data[selected_index]['Barrel%']+'</td>'
        +'<td>'+data[selected_index]['BB%']+'</td>'
        +'<td>'+data[selected_index]['K%']+'</td>'
        +'<td>'+data[selected_index]['Contact%']+'</td>'
        +'<td>'+data[selected_index]['O-Swing%']+'</td>'
        +'<td>'+data[selected_index]['HardHit%']+'</td>'
        +'<td>'+data[selected_index]['maxEV']+'</td>'
        +'<td>'+data[selected_index]['EV']+'</td>'
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
            if (v.playerid == selected) {
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
            r: [data[selected_index]['zERA'], data[selected_index]['zWHIP'], data[selected_index]['zW'], data[selected_index]['zSO'], data[selected_index]['zSv+Hld'], data[selected_index]['zERA']],
            theta: ['ERA','WHIP','W', 'SO', 'Sv+Hold', 'ERA'],
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

$(document).ready(function(){
    $("#projected_stats_table tr").hide();
    $("#projected_stats_table tr:first").show();
    $("#statcast_stats_table tr").hide();
    $("#statcast_stats_table tr:first").show();
    $.fn.z_players();
    $.fn.owners_chart('Owner', '$ Left');
    $.fn.tiers();
    $.fn.paid_histogram();
    $("input[name='playerid']").on('focusout', function(e){
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
            $("#sims").html('<font size="2">'+values+'</font>');
        });
    });
    $("#bid_form").submit(function(){
        $("#error_msg").hide();
        var player_id = $("input[name='playerid']").val();
        var bid_winner = $('input[name="owner"]:checked').val();
        var price_val = $("#price_entry").val();
        if (player_id==""){
            alert(player_id);
            $("#error_msg").text('Choose a player').show();
            return false;
        }
        if (!bid_winner){
            $("#error_msg").text('Choose a team').show();
            return false;
        }
        if (price_val<0) {
            $("#error_msg").text('Enter a price').show();
            return false;
        }
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
                var tr_id = v.playerid
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
                var tr_id = v.playerid
                $("#player_select").val(tr_id);
                $(this).create_radar_chart(tr_id);
                bid_amounts(tr_id);
                return tr_id;
            }
        });
    });

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
