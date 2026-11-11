<?php
set_time_limit(30);
ini_set('memory_limit', '256M');

$ROOT = __DIR__;
$TOKENIZER_FILE = $ROOT . '/tokenizer.json';
$CONFIG_FILE = $ROOT . '/config.json';
$WEIGHTS_FILE = $ROOT . '/model.weights';
const MODEL_NAME = 'jibay-emmbedding';
const HIDDEN = 32;
const HEADS = 2;
const FFN = 64;
const OUT_DIM = 128;
const MAX_LEN = 64;
const PAD_ID = 256;
const BOS_ID = 257;
const EOS_ID = 258;
const EN_ID = 259;
const FA_ID = 260;

function api_error($message, $type = 'invalid_request_error', $param = null, $status = 400) {
    http_response_code($status);
    header('Content-Type: application/json; charset=utf-8');
    echo json_encode(['error' => ['message' => $message, 'type' => $type, 'param' => $param, 'code' => null]], JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
    exit;
}

function byte_ids($text) {
    if ($text === '') return [];
    $v = unpack('C*', $text);
    return $v ? array_values($v) : [];
}

function detect_language($text) {
    return preg_match('/[\x{0600}-\x{06FF}\x{0750}-\x{077F}\x{08A0}-\x{08FF}]/u', $text) ? 'fa' : 'en';
}

function encode_text($text, $lang, $tokenizer) {
    $ids = byte_ids($text);
    foreach ($tokenizer['jibay']['merge_ids'] as $merge) {
        [$a, $b, $newId] = $merge;
        $next = [];
        $n = count($ids);
        for ($i = 0; $i < $n; $i++) {
            if ($i + 1 < $n && $ids[$i] === $a && $ids[$i + 1] === $b) {
                $next[] = $newId;
                $i++;
            } else {
                $next[] = $ids[$i];
            }
        }
        $ids = $next;
    }
    $count = count($ids) + 3;
    if ($count > MAX_LEN) return [null, $count];
    $langId = $lang === 'fa' ? FA_ID : EN_ID;
    return [array_merge([BOS_ID, $langId], $ids, [EOS_ID]), $count];
}

function linear_row($x, $w, $b, $in, $out) {
    $y = array_fill(0, $out, 0.0);
    for ($j = 0; $j < $out; $j++) {
        $sum = $b[$j];
        for ($i = 0; $i < $in; $i++) $sum += $x[$i] * $w[$i * $out + $j];
        $y[$j] = $sum;
    }
    return $y;
}

function layer_norm_forward($x, $gamma, $beta) {
    $n = count($x);
    $mean = array_sum($x) / $n;
    $var = 0.0;
    for ($i = 0; $i < $n; $i++) {
        $v = $x[$i] - $mean;
        $var += $v * $v;
    }
    $inv = 1.0 / sqrt($var / $n + 1e-5);
    $y = [];
    for ($i = 0; $i < $n; $i++) $y[$i] = (($x[$i] - $mean) * $inv) * $gamma[$i] + $beta[$i];
    return $y;
}

function gelu_value($x) {
    $u = 0.7978845608028654 * ($x + 0.044715 * $x * $x * $x);
    return 0.5 * $x * (1.0 + tanh($u));
}

function model_embedding($tokens, $p) {
    $d = HIDDEN;
    $f = FFN;
    $o = OUT_DIM;
    $dh = intdiv($d, HEADS);
    $t = count($tokens);
    $x = $u = $q = $k = $v = [];
    for ($i = 0; $i < $t; $i++) {
        $id = (int)$tokens[$i];
        $row = [];
        for ($j = 0; $j < $d; $j++) $row[$j] = $p['E'][$id * $d + $j] + $p['P'][$i * $d + $j];
        $x[$i] = $row;
        $u[$i] = layer_norm_forward($row, $p['ln1g'], $p['ln1b']);
        $q[$i] = linear_row($u[$i], $p['Wq'], $p['bq'], $d, $d);
        $k[$i] = linear_row($u[$i], $p['Wk'], $p['bk'], $d, $d);
        $v[$i] = linear_row($u[$i], $p['Wv'], $p['bv'], $d, $d);
    }
    $ctx = array_fill(0, $t, array_fill(0, $d, 0.0));
    $scale = 1.0 / sqrt($dh);
    for ($h = 0; $h < HEADS; $h++) {
        $start = $h * $dh;
        for ($i = 0; $i < $t; $i++) {
            $scores = [];
            $maxScore = -INF;
            for ($j = 0; $j < $t; $j++) {
                $score = 0.0;
                for ($z = 0; $z < $dh; $z++) $score += $q[$i][$start + $z] * $k[$j][$start + $z];
                $scores[$j] = $score * $scale;
                if ($scores[$j] > $maxScore) $maxScore = $scores[$j];
            }
            $den = 0.0;
            for ($j = 0; $j < $t; $j++) {
                $scores[$j] = exp($scores[$j] - $maxScore);
                $den += $scores[$j];
            }
            for ($j = 0; $j < $t; $j++) {
                $prob = $scores[$j] / max($den, 1e-12);
                for ($z = 0; $z < $dh; $z++) $ctx[$i][$start + $z] += $prob * $v[$j][$start + $z];
            }
        }
    }
    $pooled = array_fill(0, $d, 0.0);
    for ($i = 0; $i < $t; $i++) {
        $attn = linear_row($ctx[$i], $p['Wo'], $p['bo'], $d, $d);
        $r = [];
        for ($j = 0; $j < $d; $j++) $r[$j] = $x[$i][$j] + $attn[$j];
        $z = layer_norm_forward($r, $p['ln2g'], $p['ln2b']);
        $a = linear_row($z, $p['W1'], $p['b1'], $d, $f);
        for ($j = 0; $j < $f; $j++) $a[$j] = gelu_value($a[$j]);
        $ff = linear_row($a, $p['W2'], $p['b2'], $f, $d);
        for ($j = 0; $j < $d; $j++) $pooled[$j] += ($r[$j] + $ff[$j]) / $t;
    }
    $raw = linear_row($pooled, $p['Wp'], $p['bp'], $d, $o);
    $norm = sqrt(array_sum(array_map(static fn($v) => $v * $v, $raw)) + 1e-12);
    for ($i = 0; $i < $o; $i++) $raw[$i] /= $norm;
    return $raw;
}

if (($_SERVER['REQUEST_METHOD'] ?? 'GET') === 'OPTIONS') {
    header('Access-Control-Allow-Origin: *');
    header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
    header('Access-Control-Allow-Headers: Content-Type, Authorization');
    http_response_code(204);
    exit;
}

header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type, Authorization');
header('Content-Type: application/json; charset=utf-8');

$method = $_SERVER['REQUEST_METHOD'] ?? 'GET';
$payload = [];
$contentType = strtolower($_SERVER['CONTENT_TYPE'] ?? '');
if ($method === 'GET') {
    $payload = $_GET;
} elseif ($method === 'POST') {
    if (strpos($contentType, 'application/json') !== false) {
        $raw = file_get_contents('php://input');
        $decoded = json_decode((string)$raw, true);
        if (!is_array($decoded)) api_error('Request body must be a valid JSON object.', 'invalid_request_error', null, 400);
        $payload = array_merge($_GET, $decoded);
    } else {
        $form = $_POST;
        if (!$form) {
            $raw = file_get_contents('php://input');
            parse_str((string)$raw, $form);
        }
        $payload = array_merge($_GET, $form);
    }
} else {
    api_error('Only GET and POST methods are supported.', 'invalid_request_error', null, 405);
}

if (!array_key_exists('input', $payload)) api_error("Missing required parameter: 'input'.", 'invalid_request_error', 'input', 400);
$model = isset($payload['model']) ? (string)$payload['model'] : MODEL_NAME;
if ($model !== MODEL_NAME) api_error('The requested model is not available.', 'invalid_request_error', 'model', 404);
$input = $payload['input'];
if (is_string($input)) $inputs = [$input];
elseif (is_array($input) && array_is_list($input)) $inputs = $input;
else api_error("'input' must be a string or an array of strings.", 'invalid_request_error', 'input', 400);
if (count($inputs) < 1 || count($inputs) > 2048) api_error("'input' must contain between 1 and 2048 items.", 'invalid_request_error', 'input', 400);
foreach ($inputs as $value) {
    if (!is_string($value) || trim($value) === '') api_error('Every input must be a non-empty string.', 'invalid_request_error', 'input', 400);
}
if (!is_file($TOKENIZER_FILE) || !is_file($CONFIG_FILE) || !is_file($WEIGHTS_FILE)) api_error('The model has not been trained yet. Run train.php until it creates model.weights.', 'server_error', null, 503);
$tokenizer = json_decode((string)file_get_contents($TOKENIZER_FILE), true);
$config = json_decode((string)file_get_contents($CONFIG_FILE), true);
$weightData = @unserialize((string)file_get_contents($WEIGHTS_FILE), ['allowed_classes' => false]);
if (!is_array($tokenizer) || !isset($tokenizer['jibay']['merge_ids']) || !is_array($config) || !is_array($weightData) || !isset($weightData['params'])) api_error('Model files are invalid or incomplete.', 'server_error', null, 500);
$params = $weightData['params'];
if (!isset($params['E']) || count($params['E']) !== ((int)$config['vocab_size'] * HIDDEN)) api_error('Model weights do not match config.json.', 'server_error', null, 500);
$encoding = isset($payload['encoding_format']) ? (string)$payload['encoding_format'] : 'float';
if (!in_array($encoding, ['float', 'base64'], true)) api_error("'encoding_format' must be 'float' or 'base64'.", 'invalid_request_error', 'encoding_format', 400);
if (isset($payload['dimensions']) && (int)$payload['dimensions'] !== OUT_DIM) api_error('This model returns exactly ' . OUT_DIM . ' dimensions.', 'invalid_request_error', 'dimensions', 400);

$data = [];
$totalTokens = 0;
foreach ($inputs as $index => $text) {
    $lang = detect_language($text);
    [$tokens, $tokenCount] = encode_text($text, $lang, $tokenizer);
    if ($tokens === null) api_error('Input at index ' . $index . ' exceeds the maximum context length of ' . MAX_LEN . ' tokens.', 'invalid_request_error', 'input', 400);
    $totalTokens += $tokenCount;
    $embedding = model_embedding($tokens, $params);
    if ($encoding === 'base64') {
        $binary = '';
        foreach ($embedding as $value) $binary .= pack('g', (float)$value);
        $embedding = base64_encode($binary);
    }
    $data[] = ['object' => 'embedding', 'index' => $index, 'embedding' => $embedding];
}

echo json_encode([
    'object' => 'list',
    'data' => $data,
    'model' => MODEL_NAME,
    'usage' => ['prompt_tokens' => $totalTokens, 'total_tokens' => $totalTokens]
], JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
