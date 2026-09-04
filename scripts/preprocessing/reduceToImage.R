# reduceToImage.R
# ---------------------------
# Load packages
# ---------------------------
# if (!requireNamespace("exactextractr", quietly = TRUE)) install.packages("exactextractr")
# if (!requireNamespace("terra", quietly = TRUE)) install.packages("terra")
# if (!requireNamespace("sf", quietly = TRUE)) install.packages("sf")
# if (!requireNamespace("stars", quietly = TRUE)) install.packages("stars")
# if (!requireNamespace("optparse", quietly = TRUE)) install.packages("optparse")
.libPaths(Sys.getenv("R_LIBS_USER"))

library(terra)
library(stars)
library(sf)
library(exactextractr)
library(optparse)


# ---------------------------
# reduceToImage() function
# ---------------------------
#' ReduceToImage
#' @author Thomas Lauber
#' @description
#' This function reduces polygons to an image, similar to Google Earth Engine's 
#' reduceToImage() function. It works similar to rasterize, but uses the
#' coverage fraction of each polygon to assign the value of the property. 
#' If the GDAL rasterize is wanted, one can define as reducer "centroid", 
#' so that the polygon covering the centroid of a pixel gets chosen. 
#' The default reducer is "mode", which assigns the value of the class that covers
#' most of the pixel. 
#' @param path Character. The path to a zarr store. This image defines the extent and location of the output pixels. 
#' @param property Character. The property whose values should be assigned to each pixel. 
#' @param reducer Character. The reducer to use.
#' @param raster_crs Character. The crs of the raster.
#' @param polygon_crs Character. The crs of the polygons.
#' @param touches Logical. Applies only if reducer = "centroid". If TRUE, all cells touched by lines or polygons are affected, 
#' not just those on the line render path, or whose center point is within the polygon.
#' @param plot Logical. If the outcome should be plotted.
#' @param output_path Character. Saves the final result to disk.
#' 
#' @return A stars object with the rasterized property.
reduceToImage <- function(image_path,
                          polygon_path,
                          property = "lnf_code",
                          reducer = "coverage_fractions",
                          raster_crs = "EPSG:32632",
                          polygon_crs = "EPSG:2056",
                          touches = FALSE,
                          plot = FALSE,
                          all_lnf_codes = NULL,
                          output_path = NULL) {
  # Detect file type and load the image
  if (grepl("\\.tif$|\\.tiff$", image_path, ignore.case = TRUE)) {
    spat <- rast(image_path)
    xy <- crds(spat, df=TRUE)
    x_name = names(xy)[1]
    y_name = names(xy)[2]
    if (!is.null(raster_crs)) {
      crs(spat) <- raster_crs
    }
  } else {
    ds <- read_mdim(image_path)
    x_name = names(dim(ds))[1]
    y_name = names(dim(ds))[2]
    spat <- rast(ds)
    if (!is.null(raster_crs)) {
      crs(spat) <- raster_crs
    }
  }
  # Get a binary mask of the image
  mask <- spat[[1]]
  values(mask) <- 1
  # Extract bounding box of zarr file
  bbox_SpatVector <- as.polygons(ext(spat))
  crs(bbox_SpatVector) <- crs(spat)
  bbox_sf <- st_as_sf(bbox_SpatVector)
  # Reproject bounding box to polygon_crs
  bbox_sf_reproj <- st_transform(bbox_sf, polygon_crs)

  # Load polygons
  polygons_subset_sf <- st_read(polygon_path, wkt_filter = st_as_text(st_geometry(bbox_sf_reproj)), quiet = TRUE)
  if (nrow(polygons_subset_sf) == 0) {
    return(NULL)
  }
  # Ensure property is numeric
  if (is.character(polygons_subset_sf[[property]])) {
    polygons_subset_sf[[property]] <- as.numeric(polygons_subset_sf[[property]])
  }

  # Rasterize
  if(reducer == "centroid"){
    # Use GDAL via terra::rasterize
    raster <- terra::rasterize(st_transform(polygons_subset_sf, crs(mask)), mask, field = property, touches = touches)
  } else {
    # Use the cover fraction of each polygon
    # Generate for each polyon a raster that contains the coverage fraction of the polygon at each pixel
    cov_list <- coverage_fraction(mask, st_transform(polygons_subset_sf, crs(mask)))
    coverage_stack <- rast(cov_list)
    names(coverage_stack) <- as.character(polygons_subset_sf[[property]])

    # Aggregate coverage fractions for polygons with the same property value
    # This ensures that if multiple polygons have the same lnf_code (or other property),
    # their coverage fractions are summed together before finding the mode
    unique_props <- unique(names(coverage_stack))
    if (length(unique_props) < nlyr(coverage_stack)) {
      # We have duplicate property values, need to aggregate
      aggregated_layers <- list()
      for (prop in unique_props) {
        matching_indices <- which(names(coverage_stack) == prop)
        if (length(matching_indices) == 1) {
          aggregated_layers[[prop]] <- coverage_stack[[matching_indices]]
        } else {
          # Sum multiple layers with the same property value
          aggregated_layers[[prop]] <- app(coverage_stack[[matching_indices]], sum)
        }
      }
      # Create new aggregated stack
      coverage_stack_agg <- rast(aggregated_layers)
      names(coverage_stack_agg) <- unique_props
      coverage_stack <- coverage_stack_agg
    }

    # Add coverage fraction of the background
    total_poly_coverage <- app(coverage_stack, sum)
    background_coverage <- 1 - total_poly_coverage
    names(background_coverage) <- "background"
    coverage_with_background <- c(background_coverage, coverage_stack)
    if(reducer == "mode"){
      # Get the property values (use unique values from coverage_stack after aggregation)
      # Note: coverage_stack now contains aggregated unique lnf_codes, not one per polygon
      id_map <- c(0, as.numeric(names(coverage_stack)))
      # Get the id of the raster with the highest coverage, incl background
      max_idx <- which.max(coverage_with_background)
      max_id <- classify(max_idx, matrix(c(1:(length(id_map)), id_map), ncol = 2))
      names(max_id) <- property
      raster <- max_id
    } else if(reducer == "coverage_fractions"){
      # New coverage_fractions reducer: return multi-band coverage fractions
      # Get all unique lnf_codes - either from parameter (passed from Python) or from full polygon file
      # This ensures consistent band structure across all tiles
      if (is.null(all_lnf_codes)) {
        # Fallback: load full polygon file (only if not provided by Python)
        # This should be avoided as it loads the full file per tile
        polygons_full <- st_read(polygon_path, quiet = TRUE)
        if (is.character(polygons_full[[property]])) {
          polygons_full[[property]] <- as.numeric(polygons_full[[property]])
        }
        all_lnf_codes <- sort(unique(polygons_full[[property]]))
      }
      # else: use the all_lnf_codes provided as parameter (efficient!)

      # Create fixed-band structure: one band per lnf_code (including background = 0)
      # This ensures consistent band structure across all tiles
      all_bands_with_bg <- c(0, all_lnf_codes)

      # Build coverage raster stack with zeros for missing lnf_codes
      coverage_layers <- list()

      # Add background band
      total_poly_coverage <- app(coverage_stack, sum)
      background_coverage <- 1 - total_poly_coverage
      coverage_layers[["0"]] <- background_coverage

      # Add bands for each lnf_code (zero-filled if not present)
      for (lnf_code in all_lnf_codes) {
        lnf_str <- as.character(lnf_code)
        if (lnf_str %in% names(coverage_stack)) {
          # lnf_code exists in this tile
          coverage_layers[[lnf_str]] <- coverage_stack[[lnf_str]]
        } else {
          # lnf_code doesn't exist in this tile - create zero-filled band
          zero_band <- mask
          values(zero_band) <- 0
          coverage_layers[[lnf_str]] <- zero_band
        }
      }

      # Create final raster stack (bands already in sorted order)
      raster <- rast(coverage_layers)
      names(raster) <- as.character(all_bands_with_bg)
    }
  }

  # Plotting
  if(plot){
    library(ggplot2)
    raster_df <- as.data.frame(raster, xy = TRUE, na.rm = TRUE)
    raster_df[[property]] <- as.factor(raster_df[[property]])
    # Plot your subset vector data and bounding box
    unique_vals <- sort(unique(raster_df[[property]]))
    random_colors <- setNames(sample(colors(), length(unique_vals)), unique_vals)
    p <- ggplot() +
      geom_raster(data = raster_df, aes(x = x, y = y, fill = .data[[property]])) +
      scale_fill_manual(values = random_colors, name = property) +
      # geom_sf(data = st_transform(polygons_subset_sf, crs(mask)), fill = NA, color = "blue", alpha = 0.5) +
      # geom_sf(data = st_transform(bbox_sf, crs(mask)), fill = NA, color = "red", linewidth = 1) +
      theme_minimal() +
      ggtitle("Rasterized Polygons")
    print(p)
  }

  # Reformat the raster back into stars object with same dims as input
  ds_out <- st_as_stars(raster)
  ds_out <- st_set_dimensions(ds_out, 1, names = x_name)
  ds_out <- st_set_dimensions(ds_out, 2, names = y_name)
  if(!is.null(output_path)){
    write_stars(ds_out, output_path)
  }
  return(ds_out)
}


# ---------------------------
# Command-line interface
# ---------------------------
if (interactive() == FALSE) {  # only run when called via Rscript
  option_list <- list(
    make_option(c("--image_path"), type="character"),
    make_option(c("--polygon_path"), type="character"),
    make_option(c("--output_path"), type="character"),
    make_option(c("--property"), type="character", default="lnf_code"),
    make_option(c("--reducer"), type="character", default="mode"),
    make_option(c("--all_lnf_codes"), type="character", default=NULL)
  )

  opt <- parse_args(OptionParser(option_list=option_list))

  # Parse all_lnf_codes from comma-separated string to numeric vector
  if (!is.null(opt$all_lnf_codes)) {
    all_lnf_codes_vec <- as.numeric(strsplit(opt$all_lnf_codes, ",")[[1]])
  } else {
    all_lnf_codes_vec <- NULL
  }

  reduceToImage(
    image_path = opt$image_path,
    polygon_path = opt$polygon_path,
    property = opt$property,
    reducer = opt$reducer,
    output_path = opt$output_path,
    all_lnf_codes = all_lnf_codes_vec
  )
}