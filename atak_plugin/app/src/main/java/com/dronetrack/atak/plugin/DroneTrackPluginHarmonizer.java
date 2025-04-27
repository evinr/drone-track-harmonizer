package com.dronetrack.atak.plugin;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.drawable.Drawable;
import com.atakmap.android.maps.MapView;
import com.atakmap.android.dropdown.DropDownMapComponent;
import com.atakmap.android.ipc.AtakBroadcast;

import com.dronetrack.atak.R;
import com.dronetrack.atak.DroneTrackPlugin;

import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedList;

import transapps.mapi.MapView.OnMapViewRestoredListener;
import transapps.maps.plugin.lifecycle.Lifecycle;
import transapps.maps.plugin.lifecycle.LifecycleListener;
import transapps.maps.plugin.tool.Tool;
import transapps.maps.plugin.tool.ToolDescriptor;
import transapps.mapi.MapComponent;
import transapps.maps.plugin.atak.ATAKPluginHolder;
import transapps.maps.plugin.atakplugin.ATAKPlugin;

public class DroneTrackPluginHarmonizer extends ATAKPlugin {

    private Context pluginContext;
    private ATAKPluginHolder plugin;
    private DroneTrackPlugin droneTrackPlugin;

    public DroneTrackPluginHarmonizer(Context context, ATAKPluginHolder plugin) {
        super(context, plugin);
        this.pluginContext = context;
        this.plugin = plugin;
    }

    @Override
    public void onStart() {
        // Register components
        droneTrackPlugin = new DroneTrackPlugin(pluginContext, plugin);
        plugin.registerMapComponent("com.dronetrack.atak.DroneTrackPlugin", droneTrackPlugin);
    }

    @Override
    public void onStop() {
        // Unregister components
        plugin.unregisterMapComponent("com.dronetrack.atak.DroneTrackPlugin");
    }

    @Override
    public void onConfigurationChanged(final Activity activity, final android.content.res.Configuration configuration) {
        super.onConfigurationChanged(activity, configuration);
    }

    @Override
    public Collection<Lifecycle> getLifecycles() {
        Collection<Lifecycle> lifecycles = super.getLifecycles();
        lifecycles.add(droneTrackPlugin);
        return lifecycles;
    }

    @Override
    public Collection<OnMapViewRestoredListener> getOnMapViewRestoredListeners() {
        Collection<OnMapViewRestoredListener> listeners = super.getOnMapViewRestoredListeners();
        listeners.add(droneTrackPlugin);
        return listeners;
    }

    @Override
    public Collection<ToolDescriptor> getTools() {
        Collection<ToolDescriptor> tools = super.getTools();
        
        // Create tool descriptor for drone tracker
        ToolDescriptor droneTracker = new ToolDescriptor("Drone Tracker") {
            @Override
            public String getDescription() {
                return "Display harmonized drone tracks from multiple sources";
            }

            @Override
            public Drawable getIcon() {
                return pluginContext.getResources().getDrawable(R.drawable.ic_drone_tracker);
            }

            @Override
            public Object createTool(AtakMapView atakMapView) {
                // Create a Tool implementation that opens the drone tracker
                return new Tool() {
                    @Override
                    public void onCreate(MapView mapView, String s) {
                        // Send broadcast to open drone tracker
                        Intent intent = new Intent();
                        intent.setAction("com.dronetrack.atak.SHOW_DRONE_TRACKER");
                        AtakBroadcast.getInstance().sendBroadcast(intent);
                    }

                    @Override
                    public void onDestroy() {
                        // Nothing to do
                    }
                };
            }
        };
        
        tools.add(droneTracker);
        return tools;
    }
}