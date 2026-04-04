package influx

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// BucketRetentionRule defines how long data is retained in an InfluxDB bucket.
type BucketRetentionRule struct {
	EverySeconds int64  `json:"everySeconds"` // 0 = infinite
	Type         string `json:"type"`         // "expire"
}

// BucketConfig holds parameters for creating or updating an InfluxDB bucket.
type BucketConfig struct {
	OrgID          string
	Name           string
	RetentionRules []BucketRetentionRule
	Description    string
}

// BucketInfo describes a bucket returned by the InfluxDB API.
type BucketInfo struct {
	ID             string
	OrgID          string
	Name           string
	Description    string
	RetentionRules []BucketRetentionRule
	CreatedAt      time.Time
	UpdatedAt      time.Time
}

// CreateBucketV2 creates a bucket using the InfluxDB v2 HTTP API.
func (c *Client) CreateBucketV2(ctx context.Context, cfg BucketConfig) (*BucketInfo, error) {
	payload := map[string]interface{}{
		"orgID":          cfg.OrgID,
		"name":           cfg.Name,
		"description":    cfg.Description,
		"retentionRules": cfg.RetentionRules,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.baseURL+"/api/v2/buckets", strings.NewReader(string(body)))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Token "+c.token)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("create bucket status %d: %s", resp.StatusCode, string(data))
	}

	var result struct {
		ID    string `json:"id"`
		OrgID string `json:"orgID"`
		Name  string `json:"name"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return &BucketInfo{
		ID:    result.ID,
		OrgID: result.OrgID,
		Name:  result.Name,
	}, nil
}

// ListBuckets returns all buckets for an organisation.
func (c *Client) ListBuckets(ctx context.Context, orgID string) ([]BucketInfo, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		c.baseURL+"/api/v2/buckets?orgID="+orgID, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Token "+c.token)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("list buckets status %d: %s", resp.StatusCode, string(data))
	}

	var result struct {
		Buckets []struct {
			ID   string `json:"id"`
			Name string `json:"name"`
		} `json:"buckets"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	out := make([]BucketInfo, len(result.Buckets))
	for i, b := range result.Buckets {
		out[i] = BucketInfo{ID: b.ID, Name: b.Name, OrgID: orgID}
	}
	return out, nil
}

// DeleteBucket deletes a bucket by ID.
func (c *Client) DeleteBucket(ctx context.Context, bucketID string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete,
		c.baseURL+"/api/v2/buckets/"+bucketID, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Token "+c.token)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent {
		data, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("delete bucket status %d: %s", resp.StatusCode, string(data))
	}
	return nil
}

// SetRetention updates the retention rule for a bucket.
func (c *Client) SetRetention(ctx context.Context, bucketID string, retentionSeconds int64) error {
	payload := map[string]interface{}{
		"retentionRules": []BucketRetentionRule{
			{EverySeconds: retentionSeconds, Type: "expire"},
		},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPatch,
		c.baseURL+"/api/v2/buckets/"+bucketID, strings.NewReader(string(body)))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Token "+c.token)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("set retention status %d: %s", resp.StatusCode, string(data))
	}
	return nil
}

// EnsureBucket creates a bucket if it does not already exist.
func (c *Client) EnsureBucket(ctx context.Context, orgID, name string, retentionDays int) error {
	buckets, err := c.ListBuckets(ctx, orgID)
	if err != nil {
		return fmt.Errorf("list buckets: %w", err)
	}
	for _, b := range buckets {
		if b.Name == name {
			return nil // already exists
		}
	}
	retentionSecs := int64(retentionDays) * 86400
	_, err = c.CreateBucketV2(ctx, BucketConfig{
		OrgID: orgID,
		Name:  name,
		RetentionRules: []BucketRetentionRule{
			{EverySeconds: retentionSecs, Type: "expire"},
		},
		Description: "Auto-created by srfm timeseries service",
	})
	return err
}

// WriteLineProtocol writes raw InfluxDB line protocol directly.
func (c *Client) WriteLineProtocol(ctx context.Context, bucket, org, data string) error {
	url := fmt.Sprintf("%s/api/v2/write?bucket=%s&org=%s&precision=ns",
		c.baseURL, bucket, org)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Token "+c.token)
	req.Header.Set("Content-Type", "text/plain; charset=utf-8")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("write status %d: %s", resp.StatusCode, string(body))
	}
	return nil
}

// MeasurementCardinality returns the series cardinality for a measurement.
func (c *Client) MeasurementCardinality(ctx context.Context, org, bucket, measurement string) (int64, error) {
	fluxQuery := fmt.Sprintf(`
import "influxdata/influxdb"
influxdb.cardinality(
  bucket: "%s",
  start: -30d,
  predicate: (r) => r._measurement == "%s"
)`, bucket, measurement)

	rows, err := c.Query(ctx, org, fluxQuery)
	if err != nil {
		return 0, err
	}
	for _, row := range rows {
		if v, ok := row["_value"]; ok {
			switch vt := v.(type) {
			case int64:
				return vt, nil
			case float64:
				return int64(vt), nil
			}
		}
	}
	return 0, nil
}

// TagValues returns all distinct values for a given tag key.
func (c *Client) TagValues(ctx context.Context, org, bucket, measurement, tagKey string) ([]string, error) {
	fluxQuery := fmt.Sprintf(`
from(bucket: "%s")
  |> range(start: -30d)
  |> filter(fn: (r) => r._measurement == "%s")
  |> keep(columns: ["%s"])
  |> distinct(column: "%s")
`, bucket, measurement, tagKey, tagKey)

	rows, err := c.Query(ctx, org, fluxQuery)
	if err != nil {
		return nil, err
	}
	seen := make(map[string]bool)
	for _, row := range rows {
		if v, ok := row[tagKey]; ok {
			if s, ok := v.(string); ok && s != "" {
				seen[s] = true
			}
		}
	}
	out := make([]string, 0, len(seen))
	for s := range seen {
		out = append(out, s)
	}
	return out, nil
}
