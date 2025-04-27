package com.dronetrack.atak;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import com.atakmap.android.maps.MapView;
import com.atakmap.android.dropdown.DropDownMapComponent;
import com.atakmap.android.overlay.DefaultMapGroupOverlay;
import com.atakmap.coremap.cot.event.CotEvent;
import com.atakmap.comms.CotServiceRemote;
import com.atakmap.comms.CotStreamListener;

import com.atak.plugins.impl.PluginLayoutInflater;
import transapps.maps.plugin.tool.ToolDescriptor;
import transapps.mapi.MapView.OnMapViewRestoredListener;
import transapps.maps.plugin.lifecycle.Lifecycle;
import transapps.mapi.MapComponent;
import transapps.maps.plugin.atak.ATAKPluginHolder;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

public class DroneTrackPlugin extends DropDownMapComponent implements Lifecycle, OnMapViewRestoredListener, 
                                                               CotStreamListener {

    public static final String TAG = "DroneTrackPlugin";
    private Context pluginContext;
    private View templateView;
    private MapView mapView;
    
    private boolean sdrLayerEnabled = true;
    private boolean djiLayerEnabled = true;
    
    private CotServiceRemote cotServiceRemote;
    private DefaultMapGroupOverlay sdrOverlay;
    private DefaultMapGroupOverlay djiOverlay;
    
    private int sdrDroneCount = 0;
    private int djiDroneCount = 0;
    private Set<String> uniqueDroneIds = new HashSet<>();

    public DroneTrackPlugin(Context context, ATAKPluginHolder plugin) {
        pluginContext = context;
        setSubscribeToEvents(true);
    }

    @Override
    public void onCreate(final Map map, Bundle bundle) {
        this.mapView = map.getMapView();
        
        // Create overlays for drone tracks
        sdrOverlay = new DefaultMapGroupOverlay(mapView, "SDR Drone Tracks");
        djiOverlay = new DefaultMapGroupOverlay(mapView, "DJI Aeroscope Tracks");
        
        // Add overlays to map
        mapView.addMapOverlay(sdrOverlay);
        mapView.addMapOverlay(djiOverlay);
        
        // Initialize CoT service
        cotServiceRemote = new CotServiceRemote();
        cotServiceRemote.connect(pluginContext);
        cotServiceRemote.registerListener(this);
        
        // Initialize UI
        PluginLayoutInflater inflater = PluginLayoutInflater.getLayoutInflater(pluginContext);
        templateView = inflater.inflate(R.layout.drone_track_layout, null);
        
        // Initialize controls
        Switch sdrSwitch = templateView.findViewById(R.id.sdrSwitch);
        Switch djiSwitch = templateView.findViewById(R.id.djiSwitch);
        TextView statsText = templateView.findViewById(R.id.statsText);
        
        // Set initial state
        sdrSwitch.setChecked(sdrLayerEnabled);
        djiSwitch.setChecked(djiLayerEnabled);
        updateStats(statsText);
        
        // Set listeners
        sdrSwitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                sdrLayerEnabled = isChecked;
                sdrOverlay.setVisible(isChecked);
                updateStats(statsText);
            }
        });
        
        djiSwitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                djiLayerEnabled = isChecked;
                djiOverlay.setVisible(isChecked);
                updateStats(statsText);
            }
        });
        
        Button refreshButton = templateView.findViewById(R.id.refreshButton);
        refreshButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                updateStats(statsText);
                Toast.makeText(pluginContext, "Refreshed drone statistics", Toast.LENGTH_SHORT).show();
            }
        });
    }
    
    private void updateStats(TextView statsText) {
        if (statsText != null) {
            statsText.setText(String.format(
                "SDR Drones: %d\nAeroscope Drones: %d\nUnique Drones: %d", 
                sdrDroneCount, djiDroneCount, uniqueDroneIds.size()));
        }
    }

    @Override
    public void onReceiveCot(CotEvent cotEvent) {
        // Check if this is a drone CoT event
        if (cotEvent == null || cotEvent.getType() == null) {
            return;
        }
        
        // Extract details
        String uid = cotEvent.getUID();
        String type = cotEvent.getType();
        String callsign = cotEvent.getCall();
        
        // Check if this is a drone event based on naming convention
        if (callsign != null && (callsign.contains("SDR_Drone") || callsign.contains("Aeroscope"))) {
            if (callsign.contains("SDR_Drone")) {
                // Add to SDR layer
                sdrDroneCount++;
                sdrOverlay.addItem(cotEvent);
            } else if (callsign.contains("Aeroscope")) {
                // Add to DJI layer
                djiDroneCount++;
                djiOverlay.addItem(cotEvent);
            }
            
            // Extract drone ID for uniqueness check
            String droneId = extractDroneId(callsign);
            if (droneId != null) {
                uniqueDroneIds.add(droneId);
            }
            
            // Update stats
            TextView statsText = templateView.findViewById(R.id.statsText);
            updateStats(statsText);
        }
    }
    
    private String extractDroneId(String callsign) {
        // Extract drone ID from callsign (e.g., "SDR_Drone_123" -> "123")
        if (callsign == null || callsign.isEmpty()) {
            return null;
        }
        
        String[] parts = callsign.split("_");
        if (parts.length >= 3) {
            return parts[parts.length - 1];
        }
        
        return null;
    }

    @Override
    public void onMapViewRestored(Collection<String> collection) {
        // Update UI when map is restored
        if (templateView != null) {
            TextView statsText = templateView.findViewById(R.id.statsText);
            updateStats(statsText);
        }
    }

    @Override
    public void onShow(String s) {
        if (templateView != null) {
            showDropDown(templateView, DropDownManager.HEIGHT_WRAP_CONTENT, 
                         DropDownManager.WIDTH_FILL_CONTENT, false, null);
        }
    }

    @Override
    public void onHide(String s) {
        closeDropDown();
    }

    @Override
    public void onDestroy() {
        if (cotServiceRemote != null) {
            cotServiceRemote.unregisterListener(this);
            cotServiceRemote.disconnect();
        }
        
        if (mapView != null) {
            mapView.removeMapOverlay(sdrOverlay);
            mapView.removeMapOverlay(djiOverlay);
        }
    }

    @Override
    public void onCreateDrawingMenu(Menu menu, MapView mapView) {
        // Not used
    }

    @Override
    public void onCreateSidebarMenu(Map map, Menu menu) {
        // Add menu item to open drone tracker
        MenuItem droneTracker = menu.add("Drone Tracker");
        droneTracker.setIcon(R.drawable.ic_drone_tracker);
        droneTracker.setOnMenuItemClickListener(new MenuItem.OnMenuItemClickListener() {
            @Override
            public boolean onMenuItemClick(MenuItem item) {
                onShow(null);
                return true;
            }
        });
    }
}